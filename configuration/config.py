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
    mask_ratio: int 
    plots: bool
    save_dir: str
    project: str
    name: str
    save_txt: bool
    conf: Optional[float]
    plots: bool
    half: bool
    save_hybrid: bool  # (bool) save hybrid version of labels (labels + additional predictions)
    iou: float
    max_det: int
    save_json: bool  # (bool) save results to JSON file

@dataclass
class PredictionSettings:
    source: Optional[str]
    augment: bool
    agnostic_nms: bool
    classes: Optional[Any]
    boxes: bool
    show: bool
    conf: Optional[float]
    visualize: bool
    line_width: Optional[int]
    show_conf: bool
    show_labels: bool 
    retina_masks: bool
    save_txt: bool
    save_conf: bool
    save_crop: bool
    save: bool
    iou: float
    max_det: int
    project: Optional[str] = None
    name: Optional[str] = "predict_results"
    verbose: bool = False


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
    overlap_mask: bool

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
    model_path: Optional[str]
    save_model_path: Optional[str]
    data: Optional[str]
    model_size: str
    augment: bool
    train: TrainSettings
    predict: PredictionSettings
    hyp: HyperParameters
    augment_config: AugmentationSettings
    dataset_config: CocoYoloConfig
    


def load_config(yaml_path: str = None) -> Config:
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "config.yaml"

    with open(yaml_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
        
    return Config(
        task=cfg_dict["task"],
        mode=cfg_dict["mode"],
        model_path=cfg_dict["model_path"],
        save_model_path = cfg_dict["save_model_path"],
        model_size = cfg_dict["model_size"],
        data = cfg_dict["data"],
        augment = cfg_dict["augment"],
        train=TrainSettings(**cfg_dict['train']),
        predict=PredictionSettings(**cfg_dict['predict']),
        hyp=HyperParameters(**cfg_dict['hyperparameters']),
        augment_config=AugmentationSettings(**cfg_dict['augmentation']),
        dataset_config=CocoYoloConfig(**cfg_dict['dataset_config']),
    )
