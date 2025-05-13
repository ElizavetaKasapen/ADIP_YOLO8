import torch.nn.functional as F
from utils.data import get_targets

def preprocess_batch_images(device, batch):
    batch['img'] = batch['img'].to(device, non_blocking=True).float() / 255
    return batch

def resize_batch_images(imgs, size=640):
    return F.interpolate(imgs, size=(size, size), mode='bilinear', align_corners=False)

def preprocess_batch(device, batch, task, size=640):
    batch = preprocess_batch_images(device, batch)

    imgs = batch['img']
    _, _, h, w = imgs.shape

    if (h, w) != (size, size):
        
        imgs = resize_batch_images(imgs, size)

        # If masks exist, resize them to match new image size
        if 'masks' in batch:
            batch['masks'] = F.interpolate(batch['masks'].unsqueeze(1).float(), size=(size, size), mode='nearest').squeeze(1)

    targets = get_targets(batch, device, task)
    return imgs, targets

