from pathlib import Path

import numpy as np
import torch

from .detect_val import DetectionValidator
from utils import processing as ops
from metrics import OKS_SIGMA, PoseMetrics, kpt_iou
from utils.plotting.plotting import output_to_target, plot_images
from utils.config import load_data_config

class PoseValidator(DetectionValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize a 'PoseValidator' object with custom parameters and assigned attributes."""
        self.args = args
        self.data, _ = load_data_config("pose")
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.metrics = PoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def preprocess(self, batch):
        """Preprocesses the batch by converting the 'keypoints' data into a float and moving it to the device."""
        batch = super().preprocess(batch)
        batch['keypoints'] = batch['keypoints'].to(self.device).float()
        return batch

    def metrics_summary_header(self):
        """Returns description of evaluation metrics in string format."""
        return ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)', 'Pose(P',
                                         'R', 'mAP50', 'mAP50-95)')

    def postprocess(self, preds):
        """Apply non-maximum suppression and return detections with high confidence scores."""
        return ops.non_max_suppression(preds,
                                       self.args.conf,
                                       self.args.iou,
                                       labels=self.lb,
                                       multi_label=True,
                                       agnostic=self.args.single_cls,
                                       max_det=self.args.max_det,
                                       nc=self.nc)

    def init_metrics(self, model):
        """Initiate pose estimation metrics for YOLO model."""
        super().init_metrics(model)
        self.kpt_shape = self.data.kpt_shape 
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            kpts = batch['keypoints'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            nk = kpts.shape[1]  # number of keypoints
            shape = batch['ori_shape'][si]
            correct_kpts = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, correct_kpts, *torch.zeros(
                        (2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                            ratio_pad=batch['ratio_pad'][si])  # native-space pred
            pred_kpts = predn[:, 6:].view(npr, nk, -1)
            ops.scale_coords(batch['img'][si].shape[1:], pred_kpts, shape, ratio_pad=batch['ratio_pad'][si])

            # Evaluate
            if nl:
                height, width = batch['img'].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device=self.device)  # target boxes
                ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                ratio_pad=batch['ratio_pad'][si])  # native-space labels
                tkpts = kpts.clone()
                tkpts[..., 0] *= width
                tkpts[..., 1] *= height
                tkpts = ops.scale_coords(batch['img'][si].shape[1:], tkpts, shape, ratio_pad=batch['ratio_pad'][si])
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn[:, :6], labelsn)
                correct_kpts = self._process_batch(predn[:, :6], labelsn, pred_kpts, tkpts)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)

            # Append correct_masks, correct_boxes, pconf, pcls, tcls
            self.stats.append((correct_bboxes, correct_kpts, pred[:, 4], pred[:, 5], cls.squeeze(-1)))

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
          

    def _process_batch(self, detections, labels, pred_kpts=None, gt_kpts=None):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
            pred_kpts (array[N, 51]), 51 = 17 * 3
            gt_kpts (array[N, 51])
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        if pred_kpts is not None and gt_kpts is not None:
            area = ops.xyxy2xywh(labels[:, 1:])[:, 2:].prod(1) * 0.53
            iou = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area)
        else:  # boxes
            iou = ops.box_iou(labels[:, 1:], detections[:, :4])
        
        labels = labels.to(detections.device)
        correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)

        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(self.iouv)):
            iou_threshold = self.iouv[i].to(iou.device)
            correct_class = correct_class.to(iou.device)
            x = torch.where((iou >= iou_threshold) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=detections.device)

    def plot_val_samples(self, batch, ni):
        """Plots and saves validation set samples with predicted bounding boxes and keypoints."""
        plot_images(batch['img'],
                    batch['batch_idx'],
                    batch['cls'].squeeze(-1),
                    batch['bboxes'],
                    kpts=batch['keypoints'],
                    paths=batch['im_file'],
                    fname=Path(self.save_dir) / Path(f'val_batch{ni}_labels.jpg'),
                    names=self.names,
                    on_plot=self.on_plot)

    def plot_predictions(self, batch, preds, ni):
        """Plots predictions for YOLO model."""
        batch_idx, cls, bboxes, _ = output_to_target(preds, self.args.max_det)
        pred_kpts = torch.cat([p[:, 6:].view(-1, *self.kpt_shape) for p in preds], 0)
        plot_images(
            images=batch['img'],
            batch_idx=batch_idx,
            cls=cls,
            bboxes=bboxes,
            masks=np.zeros(0, dtype=np.uint8),
            kpts=pred_kpts,
            paths=batch['im_file'],
            fname= Path(self.save_dir) / Path(f'val_batch{ni}_pred.jpg'),
            names=self.names,
            on_plot=self.on_plot)  # pred


    def pred_to_json(self, predn, filename):
        """Converts YOLO predictions to COCO JSON format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.json_results.append({
                'image_id': image_id,
                'category_id': self.class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'keypoints': p[6:],
                'score': round(p[4], 5)})

