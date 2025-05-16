
from configuration.config_loader import ConfigManager
from predictor import set_predictor
from model.yolo import build_yolov8
from utils.config import load_data_config

cfg = ConfigManager.get()

data_cfg, names_dict = load_data_config(cfg.task, cfg.dataset_config.out_dir)

img_path="test_predict\\bus.jpg" 
cfg.model = "checkpoints\yolov8_segment_epoch1.pt"
#cfg.model = "checkpoints\\yolov8_pose_epoch1.pt"
#cfg.model = "checkpoints\\yolov8_detect_detect_epoch2.pt"
model = build_yolov8(
    model_size=cfg.model_size,
    model_path=cfg.model_path,
    num_classes=data_cfg.nc,
    names_dict=names_dict,
    task=cfg.task,
)

predictor = set_predictor(task=cfg.task, model=model)

results = predictor(img_path) #TODO test also with img object
for r in results:
    boxes = r.boxes.data  # Boxes object for bbox outputs
    print(f"*** boxes: *** {boxes}\n")
    masks = r.masks  # Masks object for segment masks outputs
    print(f"*** masks: *** {masks}\n")
    #probs = r.probs  
    #print(f"*** probs: *** {probs}\n")
    print(f"*** path to result: *** {r.path}")