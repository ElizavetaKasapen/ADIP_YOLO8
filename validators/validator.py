from validators.detect_val import DetectionValidator
from validators.segment_val import SegmentationValidator
from validators.pose_val import PoseValidator

VALIDATORS = {
    'detect': DetectionValidator,
    'segment': SegmentationValidator,
    'pose': PoseValidator
}

def get_validator(task):
    return VALIDATORS[task]