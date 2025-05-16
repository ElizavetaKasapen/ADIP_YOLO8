from configuration.config import CocoYoloConfig
from pathlib import Path

def is_yolo_dataset_created(cfg: CocoYoloConfig, task = None) -> bool:
    base_dir = Path(cfg.out_dir)
    tasks_to_check = [task] if task else cfg.ann_paths.keys()
    for task in tasks_to_check:
        task_dir = base_dir / task
        if not (task_dir / "images" / "train").exists():
            return False
        if not (task_dir / f"labels_{task}" / "train").exists():
            return False
        if not (task_dir / f"data_{task}.yaml").exists():
            return False
    return True
