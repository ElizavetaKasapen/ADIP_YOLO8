import cv2
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from pathlib import Path
from itertools import repeat
import numpy as np
import torch
from .base import BaseDataset
from utils.config import NUM_THREADS, TQDM_BAR_FORMAT, LOCAL_RANK
from utils.io import LOGGER, get_hash, is_dir_writeable
from utils.data import verify_image_label, img2label_paths
from dataset.instances import Instances
from .augment import build_transforms, ComposeTransforms, ResizeWithPadding
from .data_formatter import Format


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        enable_masks (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        enable_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self, *args, labels_path = "labels", data=None, aug_config=None, enable_masks=False, enable_keypoints=False, augment = True, **kwargs):
        self.enable_masks = enable_masks
        self.enable_keypoints = enable_keypoints
        self.data = data
        self.aug_config = aug_config
        self.labels_path = labels_path
        self.imgsz = self.aug_config.imgsz
        self.augment = augment
        assert not (self.enable_masks and self.enable_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(*args, augment=augment, **kwargs)

    def build_label_cache(self, path=Path('./labels.cache')):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.image_paths)
        #nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        nkpt, ndim = self.data.kpt_shape if self.data.kpt_shape else (0, 0)
        if self.enable_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.image_paths, self.label_paths, repeat(self.prefix),
                                             repeat(self.enable_keypoints), repeat(len(self.data.names)), repeat(nkpt),
                                             repeat(ndim)))
            pbar = tqdm(results, desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format='xywh'))
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING No labels found in {path}.')
        x['hash'] = get_hash(self.label_paths + self.image_paths)
        x['results'] = nf, nm, ne, nc, len(self.image_paths)
        x['msgs'] = msgs  # warnings
        #x['version'] = self.cache_version  # cache version
        if is_dir_writeable(path.parent):
            if path.exists():
                path.unlink()  # remove *.cache file if exists
            np.save(str(path), x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{self.prefix}New cache created: {path}')
        else:
            LOGGER.warning(f'{self.prefix}WARNING Cache directory {path.parent} is not writeable, cache not saved.')
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_paths = img2label_paths(self.image_paths, self.labels_path)
        cache_path = Path(self.label_paths[0]).parent.with_suffix('.cache')
        try:
            import gc
            gc.disable()  
            cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True  # load dict
            gc.enable()
            #assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_paths + self.image_paths)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.build_label_cache(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=self.prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        if nf == 0:  # number of labels found
            raise FileNotFoundError(f'{self.prefix}No labels found in {cache_path}, can not start training.')

        # Read cache
        [cache.pop(k) for k in ('hash', 'msgs')] 
        labels = cache['labels']
        self.image_paths = [lb['im_file'] for lb in labels]  # update image_paths

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING  Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            raise ValueError(f'All labels empty in {cache_path}, can not start training without labels.')
        return labels

    def create_transforms(self):
        """Builds and appends transforms to the list."""
        if self.augment:
            self.aug_config.mosaic = self.aug_config.mosaic if self.augment and not self.rect else 0.0
            self.aug_config.mixup = self.aug_config.mixup if self.augment and not self.rect else 0.0
            transforms = build_transforms(self, self.imgsz, self.aug_config)
        else:
            transforms = ComposeTransforms([ResizeWithPadding(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
       # print(f"create_transforms function. transforms: {transforms}\n self.imgsz: {self.imgsz}")
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.enable_masks,
                   return_keypoint=self.enable_keypoints,
                   batch_idx=True,
                   mask_ratio=self.aug_config.mask_ratio,
                   mask_overlap=self.aug_config.overlap_mask))
        return transforms

    def disable_mosaic(self):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        self.aug_config.mosaic = 0.0  # set mosaic ratio=0.0
        self.aug_config.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.aug_config.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.create_transforms()

    def wrap_label_instances(self, label):
        """custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch
    


