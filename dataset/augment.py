import math
import random
from copy import deepcopy

import cv2
import numpy as np
from dataset.instances import Instances
from utils.io import LOGGER
from metrics.metric import bbox_ioa

class BaseAugmentation:
    """This implementation is from mmyolo."""

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        """Applies pre-processing transforms and mixup/mosaic transforms to labels data."""
        if random.uniform(0, 1) > self.p:
            return labels

        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic or MixUp
        additional_images = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(additional_images):
                additional_images[i] = self.pre_transform(data)
        labels['additional_images'] = additional_images

        # Mosaic or MixUp
        labels = self.apply_augmentation(labels)
        labels.pop('additional_images', None)
        return labels

    def apply_augmentation(self, labels):
        """Applies MixUp or Mosaic augmentation to the label dictionary."""
        raise NotImplementedError

    def get_indexes(self):
        """Gets a list of shuffled indexes for mosaic augmentation."""
        raise NotImplementedError


class Mosaic(BaseAugmentation):
    """
    Mosaic augmentation.

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
    The augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        image_size (int, optional): Image size (height and width) after mosaic pipeline of a single image. Default to 640.
        p (float, optional): Probability of applying the mosaic augmentation. Must be in the range 0-1. Default to 1.0.
        n (int, optional): The grid size, either 4 (for 2x2) or 9 (for 3x3).
    """

    def __init__(self, dataset, image_size=640, p=1.0, n=4):
        """Initializes the object with a dataset, image size, probability, and border."""
        assert 0 <= p <= 1.0, f'The probability should be in range [0, 1], but got {p}.'
        assert n in (4, 9), 'grid must be equal to 4 or 9.'
        super().__init__(dataset=dataset, p=p)
        self.dataset = dataset
        self.image_size = image_size
        self.border = (-image_size // 2, -image_size // 2)  # width, height
        self.n = n

    def get_indexes(self, buffer=True):
        """Return a list of random indexes from the dataset."""
        if buffer:  # select images from buffer
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def apply_augmentation(self, labels):
        """Apply mixup transformation to the input image and labels."""
        assert labels.get('rect_shape', None) is None, 'rect and mosaic are mutually exclusive.'
        assert len(labels.get('additional_images', [])), 'There are no other images for mosaic augment.'
        return self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)

    def _mosaic4(self, labels):
        """Create a 2x2 image mosaic."""
        mosaic_labels = []
        s = self.image_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        for i in range(4):
            labels_patch = labels if i == 0 else labels['additional_images'][i - 1]
            # Load image
            img = labels_patch['img']
            h, w = labels_patch.pop('resized_shape')

            # Place img in mosaic_image
            if i == 0:  # top left
                mosaic_image = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_image[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # mosaic_image[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self.merge_mosaic_labels(mosaic_labels)
        final_labels['img'] = mosaic_image
        return final_labels

    def _mosaic9(self, labels):
        """Create a 3x3 image mosaic."""
        mosaic_labels = []
        s = self.image_size
        hp, wp = -1, -1  # height, width previous
        for i in range(9):
            labels_patch = labels if i == 0 else labels['additional_images'][i - 1]
            # Load image
            img = labels_patch['img']
            h, w = labels_patch.pop('resized_shape')

            # Place img in mosaic_image
            if i == 0:  # center
                mosaic_image = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Image
            mosaic_image[y1:y2, x1:x2] = img[y1 - padh:, x1 - padw:]  # mosaic_image[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming image_size*2 mosaic size
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self.merge_mosaic_labels(mosaic_labels)

        final_labels['img'] = mosaic_image[-self.border[0]:self.border[0], -self.border[1]:self.border[1]]
        return final_labels

    @staticmethod
    def _update_labels(labels, padw, padh):
        """Update labels."""
        nh, nw = labels['img'].shape[:2]
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(nw, nh)
        labels['instances'].add_padding(padw, padh)
        return labels

    def merge_mosaic_labels(self, mosaic_labels):
        """Return labels with mosaic border instances clipped."""
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        image_size = self.image_size * 2  # mosaic image_size
        for labels in mosaic_labels:
            cls.append(labels['cls'])
            instances.append(labels['instances'])
        final_labels = {
            'im_file': mosaic_labels[0]['im_file'],
            'ori_shape': mosaic_labels[0]['ori_shape'],
            'resized_shape': (image_size, image_size),
            'cls': np.concatenate(cls, 0),
            'instances': Instances.concatenate(instances, axis=0),
            'mosaic_border': self.border}  # final_labels
        final_labels['instances'].clip(image_size, image_size)
        good = final_labels['instances'].remove_zero_area_boxes()
        final_labels['cls'] = final_labels['cls'][good]
        return final_labels
    

class MixUp(BaseAugmentation):

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_indexes(self):
        """Get a random index from the dataset."""
        return random.randint(0, len(self.dataset) - 1)

    def apply_augmentation(self, labels):
        """Applies MixUp augmentation"""
        mix_ratio = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        labels2 = labels['additional_images'][0]
        labels['img'] = (labels['img'] * mix_ratio + labels2['img'] * (1 - mix_ratio)).astype(np.uint8)
        labels['instances'] = Instances.concatenate([labels['instances'], labels2['instances']], axis=0)
        labels['cls'] = np.concatenate([labels['cls'], labels2['cls']], 0)
        return labels



class RandomPerspective:

    def __init__(self,
                 degrees=0.0,
                 translate=0.1,
                 scale=0.5,
                 shear=0.0,
                 perspective=0.0,
                 border=(0, 0),
                 pre_transform=None):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        # Mosaic border
        self.border = border
        self.pre_transform = pre_transform

    def affine_transform(self, img, border):
        """Center."""
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        transform_matrix = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (border[0] != 0) or (border[1] != 0) or (transform_matrix != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, transform_matrix, dsize=self.size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, transform_matrix[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, transform_matrix, s
    
    def __call__(self, data):
        img = data['img']
        self.size = (img.shape[1] + self.border[1] * 2, img.shape[0] + self.border[0] * 2)

        if self.pre_transform:
            data = self.pre_transform(data)

        img, M, scale = self.affine_transform(img, self.border)
        data['img'] = img
        data['warp_matrix'] = M
        data['scale'] = scale
        return data



class Albumentations:
    """YOLOv8 Albumentations class (optional, only used if package is installed)"""

    def __init__(self, p=1.0):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None
        prefix = 'albumentations: '
        try:
            import albumentations as A

            #check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f'{prefix}{e}')

    def __call__(self, labels):
        """Generates object detections and returns a dictionary with detection results."""
        im = labels['img']
        cls = labels['cls']
        if len(cls):
            labels['instances'].convert_bbox('xywh')
            labels['instances'].normalize(*im.shape[:2][::-1])
            bboxes = labels['instances'].bboxes
            if self.transform and random.random() < self.p:
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # transformed
                if len(new['class_labels']) > 0:  # skip update if no bbox in new im
                    labels['img'] = new['image']
                    labels['cls'] = np.array(new['class_labels'])
                    bboxes = np.array(new['bboxes'], dtype=np.float32)
            labels['instances'].update(bboxes=bboxes)
        return labels



class ColorJitterHSV:

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels):
        """Applies random horizontal or vertical flip to an image with a given probability."""
        img = labels['img']
        if self.hgain or self.sgain or self.vgain:
            gain_factors = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=gain_factors.dtype)
            lut_hue = ((x * gain_factors[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * gain_factors[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * gain_factors[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return labels
    
    

class RandomFlip:

    def __init__(self, p=0.5, direction='horizontal', flip_idx=None) -> None:
        assert direction in ['horizontal', 'vertical'], f'Support direction `horizontal` or `vertical`, got {direction}'
        assert 0 <= p <= 1.0

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels):
        """Resize image and padding for detection, instance segmentation, pose."""
        img = labels['img']
        instances = labels.pop('instances')
        instances.convert_bbox(format='xywh')
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        # Flip up-down
        if self.direction == 'vertical' and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
        if self.direction == 'horizontal' and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)
            # For keypoints
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        labels['img'] = np.ascontiguousarray(img)
        labels['instances'] = instances
        return labels
    


class ComposeTransforms:

    def __init__(self, transforms):
        """Initializes the Compose object with a list of transforms."""
        self.transforms = transforms

    def __call__(self, data):
        """Applies a series of transformations to input data."""
        for transform in self.transforms:
            data = transform(data)
        return data

    def append(self, transform):
        """Appends a new transform to the existing list of transforms."""
        self.transforms.append(transform)

    def tolist(self):
        """Converts list of transforms to a standard Python list."""
        return self.transforms

    def __repr__(self):
        """Return string representation of object."""
        format_string = f'{self.__class__.__name__}('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string



class CopyPasteAugment:

    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, labels):
        """Implement Copy-Paste augmentation, labels as nx5 np.array(cls, xyxy)."""
        im = labels['img']
        cls = labels['cls']
        h, w = im.shape[:2]
        instances = labels.pop('instances')
        instances.convert_bbox(format='xyxy')
        instances.denormalize(w, h)
        if self.p and len(instances.segments):
            n = len(instances)
            _, w, _ = im.shape  # height, width, channels
            mask_canvas = np.zeros(im.shape, np.uint8)

            # Calculate ioa first then select indexes randomly
            ins_flip = deepcopy(instances)
            ins_flip.fliplr(w)

            ioa = bbox_ioa(ins_flip.bboxes, instances.bboxes)  # intersection over area, (N, M)
            indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
            n = len(indexes)
            for j in random.sample(list(indexes), k=round(self.p * n)):
                cls = np.concatenate((cls, cls[[j]]), axis=0)
                instances = Instances.concatenate((instances, ins_flip[[j]]), axis=0)
                cv2.drawContours(mask_canvas, instances.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

            result = cv2.flip(im, 1)  # augment segments (flip left-right)
            i = cv2.flip(mask_canvas, 1).astype(bool)
            im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

        labels['img'] = im
        labels['cls'] = cls
        labels['instances'] = instances
        return labels



class ResizeWithPadding:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """Initialize ResizeWithPadding object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left
    
    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        resize_ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            resize_ratio = min(resize_ratio, 1.0)

        # Compute padding
        ratio = resize_ratio, resize_ratio  # width, height ratios
        new_unpad = int(round(shape[1] * resize_ratio)), int(round(shape[0] * resize_ratio))
        pad_w, pad_h = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            pad_w, pad_h = np.mod(pad_w, self.stride), np.mod(pad_h, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            pad_w, pad_h = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            pad_w /= 2  # divide padding into 2 sides
            pad_h /= 2
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (pad_w, pad_h))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)) if self.center else 0, int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)) if self.center else 0, int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, pad_w, pad_h)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels

    
def build_transforms(dataset, image_size, aug_params, stretch=False):
    """Convert images to a size suitable for YOLOv8 training."""
    pre_transform = ComposeTransforms([
        Mosaic(dataset, image_size=image_size, p=aug_params.mosaic), 
        CopyPasteAugment(p=aug_params.copy_paste),
        RandomPerspective(
            degrees=aug_params.degrees,
            translate=aug_params.translate,
            scale=aug_params.scale,
            shear=aug_params.shear,
            perspective=aug_params.perspective,
            pre_transform=None if stretch else ResizeWithPadding(new_shape=(image_size, image_size)),
        )])
    flip_idx = dataset.data.flip_idx if dataset.data.flip_idx else [] # for keypoints augmentation
    if dataset.enable_keypoints:
        kpt_shape = dataset.data.kpt_shape
        if len(flip_idx) == 0 and aug_params.fliplr > 0.0:
            aug_params.fliplr = 0.0
            LOGGER.warning("WARNING No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}')

    return ComposeTransforms([
        pre_transform,
        MixUp(dataset, pre_transform=pre_transform, p=aug_params.mixup),
        Albumentations(p=1.0),
        ColorJitterHSV(hgain=aug_params.hsv_h, sgain=aug_params.hsv_s, vgain=aug_params.hsv_v),
        RandomFlip(direction='vertical', p=aug_params.flipud),
        RandomFlip(direction='horizontal', p=aug_params.fliplr, flip_idx=flip_idx)])  # transforms