from configuration.config_loader import ConfigManager
from torch.cuda.amp import GradScaler
from utils.io import LOGGER
import torch
from torch import nn, optim
from .ema import ModelEMA
from .early_stopping import EarlyStopping
from utils.tools import one_cycle
from loss import get_loss
from utils.callbacks.base import get_default_callbacks
from .warmup import warmup
from .preprocess_batch import preprocess_batch
from validator import get_validator

import math
import os
from tqdm import tqdm

cfg = ConfigManager.get()

class Train:

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
        self.ema = ModelEMA(model)  # optional but improves val metrics
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.set_loss()
        self.set_validator()
        self.set_optimizer()
        self.set_scheduler()
        metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
        self.metrics = dict(
            zip(metric_keys, [0] * len(metric_keys))
        )  # TODO: init metrics for plot_results()?
        self.callbacks = get_default_callbacks()
        self.plot_idx = [0, 1, 2]
        if self.train_args.plots:
            self.plot_training_labels()  # TODO this function depends on task
        # TODO maybe add this in the future, if model  wasuploded, not  created
        # self.resume_training(ckpt)
        # self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        # self.run_callbacks('on_pretrain_routine_end')

    def set_loss(self):
        self.criterion = get_loss(self.task, self.model)

    def set_validator(self):
        self.validator = get_validator(self.task)(
            dataloader=self.val_loader, save_dir="val_results", args=self.train_args
        )

    def set_optimizer(self):
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

    def validate(self, epoch):
        ema_model = self.ema.ema
        self.validator.model = ema_model
        # classes = self.data_cfg.names
        # names_dict = {i: name for i, name in enumerate(classes)}
        # TODO test this
        metrics_dict, loss = self.validator(
            model=ema_model,
            classes=self.model.names,
            loss_function=self.criterion,
            task=self.task,
        )
        # print(f"metrics_dict:{metrics_dict}")
        # writer.add_scalar("val/loss", loss[0] + loss[1] + loss[2], epoch) #TODO maybe explain this loss later + Add that loss can be for different (fr for segment it returns 4, where 4th is for mask)
        # if metrics_dict: # TODO REDO it later during recording metrics
        #     writer.add_scalar("val/mAP50", metrics_dict['metrics/mAP50(B)'], epoch)
        #     writer.add_scalar("val/precision", metrics_dict['metrics/precision(B)'], epoch)
        #     writer.add_scalar("val/recall", metrics_dict['metrics/recall(B)'], epoch)

        # print(f"Epoch {epoch+1}: Val loss='val/box_loss: {loss[0]:.4f}', 'val/cls_loss': {loss[1]:.4f}, 'val/dfl_loss':{loss[2]:.4f}, mAP50={metrics_dict['metrics/mAP50(B)']:.4f}, P={metrics_dict['metrics/precision(B)']:.4f}, R={metrics_dict['metrics/recall(B)']:.4f}")
        return metrics_dict, loss

    def train(self):
        batches_per_epoch = len(self.train_loader)  # number of batches
        warmup_iters = (
            max(round(self.hparams.warmup_epochs * batches_per_epoch), 100)
            if self.hparams.warmup_epochs > 0
            else -1
        )  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.train_args.imgsz} train, {self.train_args.imgsz} val\n"
            f"Using {self.train_loader.num_workers} dataloader workers\n"
            f"Logging results to {self.train_args.save_dir}\n"
            f"Starting training for {self.epochs} epochs..."
        )

        if self.train_args.close_mosaic:
            base_idx = (self.epochs - self.train_args.close_mosaic) * batches_per_epoch
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        for epoch in range(self.epochs):  # TODO if you do resume, change here
            self.run_callbacks("on_train_epoch_start")
            self.model.train()
            self.optimizer.zero_grad()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            self.total_loss = 0
            for i, batch in enumerate(pbar):
                if i == 5:  # TODO del
                    break
                print(f"*** img_size*** :  {batch['img'].shape}")
                current_iter, self.gradient_accumulate = warmup(
                    epoch, i, batches_per_epoch, warmup_iters, self.optimizer, self.batch_size, self.lf
                )
                # Forward
                with torch.cuda.amp.autocast(self.use_amp):
                    imgs, targets = preprocess_batch(
                        self.device, batch, self.task, self.train_args.imgsz
                    )
                    outputs = self.model(imgs)
                    self.loss, self.loss_items = self.criterion(outputs, targets)
                    self.total_loss = (
                        (self.total_loss * i + self.loss_items) / (i + 1)
                        if self.total_loss is not None
                        else self.loss_items
                    )
                print(f"Batch [{i}]:  loss: {self.loss}, loss items: {self.loss_items}")
                # Backward
                self.scaler.scale(self.loss).backward()
                # Optimize
                if current_iter - last_opt_step >= self.gradient_accumulate:
                    self.optimizer_step()
                    last_opt_step = current_iter
                # TODO add some logs maybe
                self.scheduler.step()
                val_metrics_dict, val_loss = self.validate(
                    epoch
                )  # TODO it should be outside the epoch, this one is just for test
                print(
                    f"Validation loss: {val_loss}; validation loss items: {val_metrics_dict}"
                )  # TODO make it prettier - use tergets keys
            self.validate(
                epoch
            )  # TODO it should be outside the epoch, this one is just for test
            os.makedirs("checkpoints", exist_ok=True)
            path_to_model = f"checkpoints/yolov8_{self.task}_epoch{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "ema_state_dict": self.ema.ema.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                path_to_model,
            )

            return self.model, path_to_model

    def optimizer_step(self):  # TODO it's for the future
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

    def resume_training(self, ckpt):  # TODO is not used now
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None:
            return
        best_fitness = 0.0
        start_epoch = ckpt["epoch"] + 1
        if ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        if self.resume:
            assert start_epoch > 0, (
                f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
                f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
            )
            LOGGER.info(
                f"Resuming training from {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs"
            )
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            LOGGER.info("Closing dataloader mosaic")
            if hasattr(self.train_loader.dataset, "mosaic"):
                self.train_loader.dataset.mosaic = False
            if hasattr(self.train_loader.dataset, "close_mosaic"):
                self.train_loader.dataset.close_mosaic(hyp=self.args)

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def plot_training_labels(self):
        pass  # TODO implement later for each task

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)
