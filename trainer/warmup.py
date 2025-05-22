import numpy as np
from configuration.config_loader import ConfigManager

cfg = ConfigManager.get()

def warmup(epoch, current_iter, warmup_iters, optimizer, batch_size, lf):
    xi = [0, warmup_iters]  # x interp
    gradient_accumulate = max(1, np.interp(current_iter, xi, [1, cfg.hyp.nbs / batch_size]).round())
    for j, x in enumerate(optimizer.param_groups):
        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
        x['lr'] = np.interp(
            current_iter, xi, [cfg.hyp.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
        if 'momentum' in x:
            x['momentum'] = np.interp(current_iter, xi, [cfg.hyp.warmup_momentum, cfg.hyp.momentum])
    return current_iter, gradient_accumulate
