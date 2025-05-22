from configuration.config_loader import ConfigManager
from torch.amp import GradScaler
from utils.io import LOGGER
import torch
import shutil
from datetime import datetime
from torch import nn, optim
from .ema import ModelEMA
from .early_stopping import EarlyStopping
from utils.tools import one_cycle
from loss import get_loss, get_loss_names
from utils.callbacks import get_callbacks
from .warmup import warmup
from .preprocess_batch import preprocess_batch
from validator import get_validator

import math
import os
from tqdm import tqdm

cfg = ConfigManager.get()

class Train:
    """
        Training class for YOLOv8 models.

        Manages the training loop, optimization, validation, and logging.

        Args:
            task (str): Type of task - 'detect', 'segment', or 'pose'.
            model (nn.Module): The YOLOv8 model to be trained.
            train_loader (DataLoader): PyTorch DataLoader for training data.
            val_loader (DataLoader): PyTorch DataLoader for validation data.
            data_cfg (dict): Configuration dictionary for the dataset.

        Attributes:
            ema (ModelEMA): Exponential Moving Average wrapper for the model.
            criterion (nn.Module): Task-specific loss function.
            optimizer (torch.optim.Optimizer): Optimizer used during training.
            validator (BaseValidator): Validator object for evaluating the model.
            scaler (GradScaler): AMP gradient scaler for mixed-precision training.
            scheduler: Learning rate scheduler.
            metrics (dict): Dictionary to store evaluation metrics.
            callbacks (list): Training callbacks.
        """

    def __init__(self, task, model, train_loader, val_loader, data_cfg):
        self.task = cfg.task if cfg.task else task
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_args = cfg.train
        self.hparams = cfg.hyp
        self.batch_size = self.train_args.batch
        self.epochs = self.train_args.epochs
        self.data_cfg = data_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"
        self.model.to(self.device)
        self.ema = ModelEMA(model)  # optional but improves val metrics
        self.scaler = GradScaler(enabled=self.use_amp)
        self.set_loss()
        self.set_validator()
        self.set_optimizer()
        self.set_scheduler()
        metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
        self.metrics = dict( zip(metric_keys, [0] * len(metric_keys)))  
        self.callbacks = get_callbacks()
        self.plot_idx = [0, 1, 2]
        self.run_callbacks('on_pretrain_routine_start')
       

    def set_loss(self):
        """
        Initializes the loss function and corresponding loss name labels 
        based on the task (detection, segmentation, or pose).
        """
        self.criterion = get_loss(self.task, self.model)
        self.loss_names = get_loss_names(self.task)

    def set_validator(self):
        """
        Initializes the validation based on the task.

        The validator is used to evaluate the model's performance 
        on the validation set after each epoch.
        """
        self.validator = get_validator(self.task)(
            dataloader=self.val_loader, save_dir="val_results", args=self.train_args
        )

    def set_optimizer(self):
        """
        Configures the optimizer and computes gradient accumulation steps.

        Also scales weight decay based on batch size and nominal batch size.
        The total number of iterations is calculated for scheduler purposes.
        """
        nbs = self.hparams.nbs
        weight_decay = self.hparams.weight_decay
        self.gradient_accumulate = max(
            round(nbs / self.batch_size), 1
        )  # accumulate loss before optimizing
        weight_decay = (
            weight_decay * self.batch_size * self.gradient_accumulate / nbs
        )  # scale weight_decay
        iterations = (
            math.ceil(len(self.train_loader.dataset) / max(self.batch_size, nbs))
            * self.epochs
        )
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.train_args.optimizer,
            lr=self.hparams.lr0,
            momentum=self.hparams.momentum,
            decay=weight_decay,
            iterations=iterations,
        )

    def build_optimizer(
        self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5
    ):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate,
        momentum, weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(
            v for k, v in nn.__dict__.items() if "Norm" in k
        )  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(
                0.002 * 5 / (4 + nc), 6
            )  # lr0 fit equation to 6 decimal places
            name, lr, momentum = (
                ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            )
            self.hparams.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in ("Adam", "Adamax", "AdamW", "NAdam", "RAdam"):
            optimizer = getattr(optim, name, optim.Adam)(
                g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
            )
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
            )

        optimizer.add_param_group(
            {"params": g[0], "weight_decay": decay}
        )  # add g0 with weight_decay
        optimizer.add_param_group(
            {"params": g[1], "weight_decay": 0.0}
        )  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"optimizer: {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer

    def set_scheduler(self):
        """
        Sets the learning rate scheduler and early stopping mechanism.

        Uses a cosine or linear learning rate schedule depending on configuration.
        Initializes an early stopping utility to monitor training progress.
        """
        lrf = self.hparams.lrf
        if self.train_args.cos_lr:
            self.lf = one_cycle(1, lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - lrf) + lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = (
            EarlyStopping(patience=self.train_args.patience),
            False,
        )

    def validate(self):
        """
        Runs the validation loop using the Exponential Moving Average (EMA) model.

        Evaluates the current state of the model on the validation set and returns
        the computed metrics and validation loss.
        Args:
            None
        Returns:
            Tuple[dict, torch.Tensor]: A dictionary of validation metrics and the validation loss tensor."""
        metrics_dict, loss = self.validator(
            model=self.ema.ema,
            classes=self.model.names,
            loss_function=self.criterion,
            task=self.task,
        )
        return metrics_dict, loss


    def save_model(self, model_path):
        """
        Saves the current model, EMA model, and optimizer state to a file.

        Ensures the destination directory exists before saving the model checkpoint.

        Args:
        model_path (str): The full path (including filename) where the model should be saved."""
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "ema_state_dict": self.ema.ema.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                model_path,
            )
    
    def terminal_separator(self):
        """Prints a horizontal separator line in the terminal."""
        terminal_width = shutil.get_terminal_size().columns
        tqdm.write("-"*terminal_width)
        
    def format_loss_items(self, loss_items, prefix):
        """Formats labeled loss items as a human-readable string."""
        formatted_loss = ""
        items = self.label_loss_items(loss_items, prefix=prefix)
        for k, v in items.items():  formatted_loss += f"{k}: {v}\n"
        return formatted_loss


    def train(self, model_path = ""):
        """
        Executes the main training loop for the model.

        Handles epoch and batch iterations, gradient accumulation, learning rate scheduling,
        validation, logging, and model checkpoint saving.

        Args:
            model_path (str, optional): Optional path to save the final model. Defaults to an empty string.

        Returns:
            Tuple[nn.Module, str]: The trained model and the path to the model (model_path if provided, otherwise last chackpoint path)
        """
        batches_per_epoch = len(self.train_loader)  # number of batches
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        warmup_iters = (
            max(round(self.hparams.warmup_epochs * batches_per_epoch), 100)
            if self.hparams.warmup_epochs > 0
            else -1
        )  # number of warmup iterations
        last_opt_step = -1
        LOGGER.info(
            f"Image sizes {self.train_args.imgsz} train, {self.train_args.imgsz} val\n"
            f"Using {self.train_loader.num_workers} dataloader workers\n"
            f"Logging results to {self.train_args.save_dir}\n"
            f"Starting training for {self.epochs} epochs..."
        )

        if self.train_args.close_mosaic:
            base_idx = (self.epochs - self.train_args.close_mosaic) * batches_per_epoch
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        for epoch in range(self.epochs):  
            self.epoch = epoch
            self.model.train()
            self.optimizer.zero_grad()
            self.terminal_separator()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", dynamic_ncols=True, leave=True)
            self.running_loss = 0
            self.running_loss_items = None
            total_images = 0
            for i, batch in enumerate(pbar):
                if i ==1: break #TODO del
                total_images += batch['img'].shape[0]
                current_iter = i + batches_per_epoch * epoch
                if current_iter <= warmup_iters:
                    current_iter, self.gradient_accumulate = warmup(
                        epoch, current_iter, warmup_iters, self.optimizer, self.batch_size, self.lf
                    )
                # Forward
                with torch.amp.autocast(self.device.type):
                    imgs, targets = preprocess_batch(
                        self.device, batch, self.task, self.train_args.imgsz
                    )
                   
                    outputs = self.model(imgs)
                    modif_batch = targets
                    modif_batch['img'] = imgs
                    self.loss, self.loss_items = self.criterion(outputs, modif_batch) 
                    self.running_loss += self.loss.item()
        
                    if self.running_loss_items is None:
                        self.running_loss_items = self.loss_items
                    else:
                        self.running_loss_items = [x + y for x, y in zip(self.running_loss_items, self.loss_items)]
                   
                # Backward
                self.scaler.scale(self.loss).backward()
                # Optimize
                if current_iter - last_opt_step >= self.gradient_accumulate:
                    self.optimizer_step()
                    last_opt_step = current_iter
               # self.run_callbacks('on_batch_end') 
            self.scheduler.step()
            self.total_loss = self.running_loss / total_images 
            self.loss_items = [x / total_images  for x in self.running_loss_items]
            print(f"\nTrain loss epoch[{epoch+1}]:  \nloss: {round(float(self.total_loss), 5)}, \nloss items: {self.format_loss_items(self.loss_items, f'train')}") 
            self.metrics, self.val_loss = self.validate()  
            print(f"\nValidation loss epoch[{epoch+1}]: \n{self.format_loss_items(self.val_loss, f'val')}")  
            self.run_callbacks('on_fit_epoch_end')
            os.makedirs("checkpoints", exist_ok=True)
            path_to_model = f"checkpoints/yolov8_{self.task}_{timestamp}_epoch{epoch+1}.pt"
            self.save_model(path_to_model)
        if model_path:
            self.save_model(model_path)
            path_to_model = model_path
        return self.model, path_to_model

    def optimizer_step(self):  
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=10.0
        )  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
                
    
    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        callback = self.callbacks.get(event)
        callback(self)
