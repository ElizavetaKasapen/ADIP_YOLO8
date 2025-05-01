# TODO combine all configs + train/valid/predict part here + maybe dataset process and mb downloading it
from configuration.config_loader import ConfigManager
from utils.load_data_config import load_data_config
from create_dataset import create_yolo_dataset_from_config
from model.yolo import build_yolov8
from dataset.check import is_yolo_dataset_created
#from train import train
import os
from torch.utils.data import DataLoader
from dataset.yolo_dataset import YOLODataset
from train.train_object import Train


cfg = ConfigManager.get()

# TODO maybe put somehwere else


def get_loaders(train_set, val_set, batch_size):
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=YOLODataset.collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, collate_fn=YOLODataset.collate_fn)
    return train_loader, val_loader


# Checks if there is dataset for current task, if you don't pass the task, checks for every task
if not is_yolo_dataset_created(cfg.dataset_config, cfg.task):
    print("YOLO dataset not found. Creating it...")
    # Creates dataset for current task, if you don't have it, if you don't pass the task, checks for every task
    create_yolo_dataset_from_config(cfg.dataset_config, cfg.task)
else:
    print("YOLO dataset already exists.")

data_cfg = load_data_config(cfg.task, cfg.dataset_config.out_dir)

# if there is path to model - cfg.model - it will download weights from file
# get model
model = build_yolov8(model_size=cfg.model_size,
                     model_path=cfg.model, num_classes=data_cfg.nc, task=cfg.task)
# type('Args', (object,), cfg.hyp)() #TODO check if I need all cfg, or only part of it?
model.args = cfg.hyp
# get data
train_path = os.path.join(data_cfg.path, data_cfg.train)
val_path = os.path.join(data_cfg.path, data_cfg.val)
# get dataset and loaders #TODO set the train obj later
batch_size = cfg.train.batch
enable_masks = cfg.task == "segment"
enable_keypoints = cfg.task == "pose"
train_set = YOLODataset(img_path=train_path, labels_path=data_cfg.labels, data=data_cfg,
                        aug_config=cfg.augment, enable_masks=enable_masks, enable_keypoints=enable_keypoints, augment=True)
val_set = YOLODataset(img_path=val_path, labels_path=data_cfg.labels, data=data_cfg,
                      aug_config=cfg.augment, enable_masks=enable_masks, enable_keypoints=enable_keypoints, augment=False)
train_loader, val_loader = get_loaders(train_set, val_set, batch_size)
print(
    f"Dataloaders ready! train_loader: {len(train_loader)}; val_loader: {len(val_loader)}")

train = Train(cfg.task, model=model, train_loader = train_loader, val_loader = val_loader, train_args=cfg.train)

# TODO call train
#train()
# TODO call val
# TODO call predict
