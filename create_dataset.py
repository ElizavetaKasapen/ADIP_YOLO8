from pathlib import Path
from dataset.coco_to_yolo import create_yolo_dataset_from_coco
from configuration.config_loader import ConfigManager
from configuration.config import CocoYoloConfig

cfg = ConfigManager.get()

def create_yolo_dataset_from_config(config: CocoYoloConfig, task = None):
    tasks_to_process = [task] if task else config.ann_paths.keys()
    for task in tasks_to_process:
        ann_file = config.ann_paths[task]
        ann_path = Path(config.coco_dataset_path) / config.coco_ann_path / ann_file
        img_dir = Path(config.coco_dataset_path) / config.img_dir
        out_dir = Path(config.out_dir) / task

        create_yolo_dataset_from_coco(
            ann_path=str(ann_path),
            img_dir=str(img_dir),
            out_dir=str(out_dir),
            task=task,
            max_images=config.max_images,
            train_ratio=config.train_ratio
        )
