import math
import torch.nn as nn
from copy import deepcopy



def copy_model_attributes(a, b, include=(), exclude=()):
    """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """Updated Exponential Moving Average (EMA).
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Create EMA."""
        self.ema = deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            decay_factor  = self.decay(self.updates)

            model_state = model.state_dict()  # model state_dict 
            #model_state = de_parallel(model).state_dict()  # model state_dict
            for param_name, param_tensor in self.ema.state_dict().items():
                if param_tensor.dtype.is_floating_point:  # true for FP16 and FP32
                    param_tensor *= decay_factor
                    param_tensor += (1 - decay_factor) * model_state[param_name].detach()
                    

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_model_attributes(self.ema, model, include, exclude)

