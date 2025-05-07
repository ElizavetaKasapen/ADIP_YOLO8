import os
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO
from PIL import Image
from configuration.config_loader import ConfigManager
from configuration.config import CocoYoloConfig

cfg = ConfigManager.get()

def make_dirs(base, task):
    images_path = "images" 
    labels_path =  f"labels_{task}"
    os.makedirs(os.path.join(base, images_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(base, images_path, "val"), exist_ok=True)
    os.makedirs(os.path.join(base, labels_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(base, labels_path, "val"), exist_ok=True)
    return images_path, labels_path

def normalize_box(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return [cx, cy, w / img_w, h / img_h]

def normalize_points(points, img_w, img_h):
    return [x / img_w if i % 2 == 0 else x / img_h for i, x in enumerate(points)]


def resize_and_save_image(image_path, target_size=(640, 640), output_dir=None):
    """
    Resizes an image to the target size and saves it.
    Returns the new width and height of the image.
    """
    with Image.open(image_path) as img:
        img_resized = img.resize(target_size)
        if output_dir:
            img_resized.save(os.path.join(output_dir, os.path.basename(image_path)))  # Save resized image
        return img_resized.size  # Return the new dimensions (width, height)

    

def convert_detection(coco, image_ids, split, out_dir, img_dir, cat_id_to_index):
    label_dir = os.path.join(out_dir, f"labels_detect", split)
    image_out_dir = os.path.join(out_dir, "images", split)

    for img_id in tqdm(image_ids, desc=f"Processing {split} detection"):
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
  
        img_info["width"], img_info["height"] =  resize_and_save_image(image_path = img_path, output_dir=image_out_dir)
    
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        label_path = os.path.join(label_dir, os.path.splitext(img_info["file_name"])[0] + ".txt")

        with open(label_path, "w") as f:
            for ann in anns:
                class_id = cat_id_to_index[ann['category_id']]
                bbox = normalize_box(ann["bbox"], img_info["width"], img_info["height"])
                f.write(f"{class_id} {' '.join(f'{v:.6f}' for v in bbox)}\n")

def convert_segmentation(coco, image_ids, split, out_dir, img_dir, cat_id_to_index):
    label_dir = os.path.join(out_dir, f"labels_segment", split)
    image_out_dir = os.path.join(out_dir, "images", split)

    for img_id in tqdm(image_ids, desc=f"Processing {split} segmentation"):
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        img_info["width"], img_info["height"] =  resize_and_save_image(image_path = img_path, output_dir=image_out_dir)

        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        label_path = os.path.join(label_dir, os.path.splitext(img_info["file_name"])[0] + ".txt")

        with open(label_path, "w") as f:
            for ann in anns:
                if not ("segmentation" in ann and isinstance(ann["segmentation"], list)):
                    continue
                class_id = cat_id_to_index[ann['category_id']]
                seg = normalize_points(ann["segmentation"][0], img_info["width"], img_info["height"])
                f.write(f"{class_id} {' '.join(f'{v:.6f}' for v in seg)}\n")

def convert_pose(coco, image_ids, split, out_dir, img_dir):
    label_dir = os.path.join(out_dir, f"labels_pose", split)
    image_out_dir = os.path.join(out_dir, "images", split)

    for img_id in tqdm(image_ids, desc=f"Processing {split} pose"):
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        img_info["width"], img_info["height"] =  resize_and_save_image(image_path = img_path, output_dir=image_out_dir)

        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        label_path = os.path.join(label_dir, os.path.splitext(img_info["file_name"])[0] + ".txt")

        with open(label_path, "w") as f:
            for ann in anns:
                if ann["num_keypoints"] == 0:
                    continue

                bbox = ann["bbox"]
                bbox_normalized = normalize_box(bbox, img_info['width'], img_info['height'])
                
                kp = ann["keypoints"]
                norm_kp = [f"{kp[i] / img_info['width']:.6f} {kp[i+1] / img_info['height']:.6f} {kp[i+2]}" for i in range(0, len(kp), 3)]

                # Final line: class bbox kp1 kp2 kp3 ...
                f.write(f"0 {' '.join(map(str, bbox_normalized))} {' '.join(norm_kp)}\n")



def write_yaml(save_path, task, classes=None, nc=1, labels_path = "labels"):
    file = os.path.join(save_path, f"data_{task}.yaml")
    with open(file, "w") as f:
        f.write(f"path: {save_path}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"labels: {labels_path}\n")
        f.write(f"nc: {nc}\n")
        if task == "pose":
            f.write("kpt_shape: [17, 3]\n")
            f.write("names: ['person']\n")
            f.write("flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n")
        else:
            f.write(f"names: {classes}\n")

def create_yolo_dataset_from_coco(ann_path, img_dir, out_dir, task="detect", max_images=500, train_ratio=0.8):
    _, labels_path = make_dirs(out_dir, task)
    coco = COCO(ann_path)

    image_ids = list(coco.imgs.keys())[:max_images]
    split_idx = int(len(image_ids) * train_ratio)
    train_ids, val_ids = image_ids[:split_idx], image_ids[split_idx:]

    if task in ["detect", "segment"]:
        cats = coco.loadCats(coco.getCatIds())
        cats = sorted(cats, key=lambda x: x['id'])
        classes = [c['name'] for c in cats]
        cat_id_to_index = {cat['id']: i for i, cat in enumerate(cats)}

        if task == "detect":
            convert_detection(coco, train_ids, "train", out_dir, img_dir, cat_id_to_index)
            convert_detection(coco, val_ids, "val", out_dir, img_dir, cat_id_to_index)
        else:
            convert_segmentation(coco, train_ids, "train", out_dir, img_dir, cat_id_to_index)
            convert_segmentation(coco, val_ids, "val", out_dir, img_dir, cat_id_to_index)

        write_yaml(out_dir, task, classes=classes, nc=len(classes), labels_path=labels_path)

    elif task == "pose":
        convert_pose(coco, train_ids, "train", out_dir, img_dir)
        convert_pose(coco, val_ids, "val", out_dir, img_dir)
        write_yaml(out_dir, "pose", labels_path=labels_path)



def create_yolo_dataset(config: CocoYoloConfig, task = None):
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
