from dataclasses import dataclass
import yaml
from typing import Optional, Any
from pathlib import Path

@dataclass
class TrainSettings:
    epochs: int
    patience: int
    batch: int
    imgsz: int
    save: bool
    cache: bool
    workers: int
    pretrained: bool
    optimizer: str
    verbose: bool
    seed: int
    deterministic: bool
    single_cls: bool
    rect: bool
    cos_lr: bool
    close_mosaic: int
    resume: bool
    overlap_mask: bool
    mask_ratio: int 

@dataclass
class ValidationSettings:
    val: bool
    split: str
    conf: float
    iou: float
    max_det: int
    half: bool

@dataclass
class PredictionSettings:
    source: Optional[str]
    augment: bool
    agnostic_nms: bool
    classes: Optional[Any]
    boxes: bool

@dataclass
class SegmentationSettings:
    overlap_mask: bool
    mask_ratio: int #TODO if not needed for train func, del here

@dataclass
class HyperParameters:
    lr0: float
    lrf: float
    momentum: float
    weight_decay: float
    warmup_epochs: float
    warmup_momentum: float
    warmup_bias_lr: float
    box: float
    cls: float
    dfl: float
    pose: float
    kobj: float
    nbs: int

@dataclass
class AugmentationSettings:
    hsv_h: float
    hsv_s: float
    hsv_v: float
    degrees: float
    translate: float
    scale: float
    shear: float
    perspective: float
    flipud: float
    fliplr: float
    mosaic: float
    mixup: float
    imgsz: int
    copy_paste: float
    mask_ratio: int
    overlap_mask: bool

@dataclass
class CocoYoloConfig:
    coco_dataset_path: str
    coco_ann_path: str = "annotations"
    ann_paths: dict = None  # e.g. {"detection": "instances_val2017.json", "pose": "person_keypoints_val2017.json"}
    img_dir: str = "images"
    out_dir: str = "yolo_dataset"
    max_images: int = 500
    train_ratio: float = 0.8

@dataclass
class Config:
    task: str
    mode: str
    model: Optional[str]
    data: Optional[str]
    model_size: str
    train: TrainSettings
    val: ValidationSettings
    predict: PredictionSettings
    seg: SegmentationSettings
    hyp: HyperParameters
    augment: AugmentationSettings
    dataset_config: CocoYoloConfig
    


def load_config(yaml_path: str = None) -> Config:
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "config.yaml"

    with open(yaml_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
        
    return Config(
        task=cfg_dict["task"],
        mode=cfg_dict["mode"],
        model=cfg_dict["model"],
        model_size = cfg_dict["model_size"],
        data = cfg_dict["data"],
        train=TrainSettings(**{k: cfg_dict[k] for k in TrainSettings.__annotations__.keys()}),
        val=ValidationSettings(**{k: cfg_dict[k] for k in ValidationSettings.__annotations__.keys()}),
        predict=PredictionSettings(**{k: cfg_dict[k] for k in PredictionSettings.__annotations__.keys()}),
        seg=SegmentationSettings(**{k: cfg_dict[k] for k in SegmentationSettings.__annotations__.keys()}),
        hyp=HyperParameters(**{k: cfg_dict[k] for k in HyperParameters.__annotations__.keys()}),
        augment=AugmentationSettings(**{k: cfg_dict[k] for k in AugmentationSettings.__annotations__.keys() if k in cfg_dict}),
        dataset_config = CocoYoloConfig(**{k: cfg_dict[k] for k in CocoYoloConfig.__annotations__.keys() if k in cfg_dict}),
    )
