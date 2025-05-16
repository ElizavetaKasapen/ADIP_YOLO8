import torch


def get_targets(batch, device, task):
    task_keys = {
        'detect': ['cls', 'bboxes', 'batch_idx'], 
        'segment': ['cls', 'bboxes', 'batch_idx', 'masks'],
        'pose': ['cls', 'bboxes', 'batch_idx', 'keypoints']
    }

    if task not in task_keys:
        raise ValueError(f"Unsupported task: {task}. Supported tasks: {list(task_keys.keys())}")

    targets = {
        k: v.to(device)
        for k, v in batch.items()
        if k in task_keys[task] and isinstance(v, torch.Tensor)
    }
    if 'cls' in targets:
        targets['cls'] = targets['cls'].long()
    if 'keypoints' in targets:
        targets['keypoints'] = targets['keypoints'].float()
    return targets