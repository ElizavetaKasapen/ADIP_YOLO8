from validator.detect_val import DetectionValidator
from validator.segment_val import SegmentationValidator
from validator.pose_val import PoseValidator

VALIDATORS = {
    'detect': DetectionValidator,
    'segment': SegmentationValidator,
    'pose': PoseValidator
}

def get_validator(task):
    return VALIDATORS[task]