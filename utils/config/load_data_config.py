from pathlib import Path
import yaml
from configuration.yolo_data_config import  YoloDataConfig

def load_data_config(task: str, out_dir: str = "yolo_dataset") -> dict:
    yaml_path = Path(out_dir) / task / f"data_{task}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Expected data config at {yaml_path}, but it was not found.")

    with open(yaml_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    data_config = YoloDataConfig(**raw_cfg)
    classes = data_config.names
    names_dict = {i: name for i, name in enumerate(classes)}
    return data_config, names_dict

