from torch.utils.data import DataLoader
from .yolo_dataset import YOLODataset


def create_datasets(data_cfg, cfg, train_path, val_path):
    enable_masks = cfg.task == "segment"
    enable_keypoints = cfg.task == "pose"
    
    train_set = YOLODataset(
        img_path=train_path,
        labels_path=data_cfg.labels,
        data=data_cfg,
        aug_config=cfg.augment_config,
        enable_masks=enable_masks,
        enable_keypoints=enable_keypoints,
        augment=True,
    )
    val_set = YOLODataset(
        img_path=val_path,
        labels_path=data_cfg.labels,
        data=data_cfg,
        aug_config=cfg.augment_config,
        enable_masks=enable_masks,
        enable_keypoints=enable_keypoints,
        augment=False,
    )
    return train_set, val_set


def get_loaders(train_set, val_set, batch_size):
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=YOLODataset.collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=YOLODataset.collate_fn
    )
    return train_loader, val_loader