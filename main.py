from configuration.config_loader import ConfigManager
from utils.config import load_data_config
from model.yolo import build_yolov8
from utils.data import is_yolo_dataset_created
from dataset.loader import get_loaders, create_datasets
from dataset.coco_to_yolo import create_yolo_dataset
from predictor import set_predictor
import os

# Load global configuration
cfg = ConfigManager.get()

# Check if the YOLO dataset for the given task exists; if not, create it
if not is_yolo_dataset_created(cfg.dataset_config, cfg.task):
    print("YOLO dataset not found. Creating it...")
    create_yolo_dataset(cfg.dataset_config, cfg.task)
else:
    print("YOLO dataset already exists.")

# Load data configuration and class names
data_cfg, names_dict = load_data_config(cfg.task, cfg.dataset_config.out_dir)

# Build YOLOv8 model
model = build_yolov8(
    model_size=cfg.model_size,
    model_path=cfg.model_path,
    num_classes=data_cfg.nc,
    names_dict=names_dict,
    task=cfg.task,
    args=cfg.hyp,
)

# Handle training or prediction mode
if cfg.mode == "train":
    from trainer.train_object import Train

    # Build paths to training and validation datasets
    train_path = os.path.join(data_cfg.path, data_cfg.train)
    val_path = os.path.join(data_cfg.path, data_cfg.val)

    # Create dataset and data loaders
    batch_size = cfg.train.batch
    train_set, val_set = create_datasets(data_cfg, cfg, train_path, val_path)
    train_loader, val_loader = get_loaders(train_set, val_set, batch_size)
    print(f"Dataloaders ready! train_loader: {len(train_loader)}; val_loader: {len(val_loader)}")

    # Initialize trainer
    trainer = Train(
        task=cfg.task,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        data_cfg=data_cfg,
    )

    # Train model and save checkpoint
    if cfg.save_model_path is None:
        model, path_to_model = trainer.train()  # Returns the path to the last checkpoint
    else:
        model, path_to_model = trainer.train(model_path=cfg.save_model_path)

    print(f"Your model is trained! You can find it here: {path_to_model}")

elif cfg.mode == "predict":
    # Initialize predictor and run inference
    predictor = set_predictor(task=cfg.task, model=model)
    results = predictor(cfg.predict.source)

    for r in results:
        boxes = r.boxes.data  # Tensor of bounding box predictions
        print(f"boxes: {boxes}\n")

        masks = r.masks  # Mask predictions (only for segmentation)
        print(f"masks:  {masks}\n")

        print(f"path to result: {r.path}")

else:
    raise ValueError(f"Unsupported mode: {cfg.mode}. Please use 'train' or 'predict'.")
