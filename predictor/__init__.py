from .predict_detect import DetectionPredictor
from .predict_segment  import SegmentationPredictor
from .predict_pose import PosePredictor

PREDICTORS = {
    'detect': DetectionPredictor,
    'segment': SegmentationPredictor,
    'pose': PosePredictor
}

def set_predictor(task, model = None):
    return PREDICTORS[task](model=model, task = task)