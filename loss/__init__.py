from .detect_loss import DetectionLoss
from .segment_loss import SegmentationLoss
from .pose_loss import PoseLoss


LOSS = {"detect": DetectionLoss, "segment": SegmentationLoss, "pose": PoseLoss}

detect_names = 'box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss'
segment_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss'
pose_names = 'box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss'
LOSS_NAMES = {"detect": detect_names, "segment": segment_names, "pose": pose_names}

def get_loss(task, model):
    return LOSS[task](model)

def get_loss_names(task):
    return LOSS_NAMES[task]
