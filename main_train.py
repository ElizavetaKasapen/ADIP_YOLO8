from configuration.config_loader import ConfigManager
from utils.config import load_data_config
from model.yolo import build_yolov8
from utils.data import is_yolo_dataset_created
from dataset.loader import get_loaders, create_datasets
from dataset.coco_to_yolo import create_yolo_dataset

# from train import train
import os
from train.train_object import Train

cfg = ConfigManager.get()

# Checks if there is dataset for current task, if you don't pass the task, checks for every task
if not is_yolo_dataset_created(cfg.dataset_config, cfg.task):
    print("YOLO dataset not found. Creating it...")
    # Creates dataset for current task, if you don't have it, if you don't pass the task, checks for every task
    create_yolo_dataset(cfg.dataset_config, cfg.task)
else:
    print("YOLO dataset already exists.")

data_cfg, names_dict = load_data_config(cfg.task, cfg.dataset_config.out_dir)

model = build_yolov8(
    model_size=cfg.model_size,
    model_path=cfg.model_path,
    num_classes=data_cfg.nc,
    names_dict=names_dict,
    task=cfg.task,
    args=cfg.hyp,
)
# get data
train_path = os.path.join(data_cfg.path, data_cfg.train)
val_path = os.path.join(data_cfg.path, data_cfg.val)

batch_size = cfg.train.batch
train_set, val_set = create_datasets(data_cfg, cfg, train_path, val_path)
train_loader, val_loader = get_loaders(train_set, val_set, batch_size)
print(
    f"Dataloaders ready! train_loader: {len(train_loader)}; val_loader: {len(val_loader)}"
)

train = Train(
    cfg.task,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    data_cfg=data_cfg,
)
if cfg.save_model_path == None:
    model, path_to_model = train.train() #returns the path of last checkpoit
else:
    model, path_to_model = train.train(model_path = cfg.save_model_path)
print(f"Your model is trained! You can find it here: {path_to_model}")


    
