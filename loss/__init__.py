from .detect_loss import DetectionLoss
from .segment_loss import SegmentationLoss
from .pose_loss import PoseLoss


LOSS = {"detect": DetectionLoss, "segment": SegmentationLoss, "pose": PoseLoss}

def get_loss(task, model):
    return LOSS[task](model)
