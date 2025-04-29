#TODO combine all configs + train/valid/predict part here + maybe dataset process and mb downloading it
from configuration.config_loader import ConfigManager
from utils.load_data_config import load_data_config
from create_dataset import create_yolo_dataset_from_config
from model.yolo import build_yolov8
from dataset.check import is_yolo_dataset_created


cfg = ConfigManager.get()

if not is_yolo_dataset_created(cfg.dataset_config, cfg.task): # Checks if there is dataset for current task, if you don't pass the task, checks for every task
    print("YOLO dataset not found. Creating it...")
    create_yolo_dataset_from_config(cfg.dataset_config, cfg.task)  # Creates dataset for current task, if you don't have it, if you don't pass the task, checks for every task
else:
    print("YOLO dataset already exists.")
    
data_cfg = load_data_config(cfg.task, cfg.dataset_config.out_dir)

# if there is path to model - cfg.model - it will download weights from file
model = build_yolov8(model_size=cfg.model_size, model_path=cfg.model, num_classes=data_cfg.nc, task=cfg.task)
#TODO call train 

#TODO call val
#TODO call predict