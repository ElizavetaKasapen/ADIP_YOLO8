from loss import DetectionLoss, SegmentationLoss, PoseLoss
from validators.detect_val import DetectionValidator
from validators.segment_val import SegmentationValidator
from validators.pose_val import PoseValidator
from configuration.config_loader import ConfigManager

cfg = ConfigManager.get()

LOSS = {
    'detect': DetectionLoss,
    'segment': SegmentationLoss,
    'pose': PoseLoss
}

VALIDATORS = {
    'detect': DetectionValidator,
    'segment': SegmentationValidator,
    'pose': PoseValidator
}

class Train():
    
    def __init__(self, task, model, train_loader, val_loader):
        self.task = cfg.task if cfg.task else task
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_args = cfg.train
        self.hparams = cfg.hyp
        self.criterion = self.get_loss(self.task, self.model)
        self.validator = self.get_validator(val_loader, model, self.train_args)
            
    def get_loss(self):
        return LOSS[self.task](self.model)
    
    def get_validator(self):
        ValidatorClass = VALIDATORS[self.task]
        validator = ValidatorClass(dataloader=self.val_loader, save_dir='val_results', args=self.train_args)
        validator.args = self.model.args
        return validator
        