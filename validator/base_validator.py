import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from utils.io import LOGGER, increment_path
from utils.config import RANK, TQDM_BAR_FORMAT  
from utils.data import get_targets
from utils.callbacks import get_callbacks

class BaseValidator:
    """
    BaseValidator

    A base class for creating validators.

    Attributes:
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        args (SimpleNamespace): Configuration for the validator.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        speed (float): Batch processing speed in seconds.
        jdict (dict): Dictionary to store validation results.
        save_dir (Path): Directory to save results.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, _callbacks=None):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
        """
        self.dataloader = dataloader
        self.progress_bar = pbar
        self.model = None
        self.device = None
        self.batch_i = None 
        self.json_results = None
        self.save_dir = save_dir
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001
        self.plots = {}

    def dir_to_save(self):
        root =  Path()
        runs_dir = str(root / "runs")
        project = self.args.project or Path(runs_dir) / self.args.task
        name = self.args.name 
        self.save_dir = self.save_dir or increment_path(Path(project) / name,
                                                   exist_ok=self.args.exist_ok if RANK in (-1, 0) else True)
        Path((self.save_dir / 'labels' if self.args.save_txt else self.save_dir)).mkdir(parents=True, exist_ok=True)

    def __call__(self, model=None, classes = 80, loss_function = None, task='detect'): 
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        self.args.task = task
        self.loss_function = loss_function
        assert model is not None, 'Either trainer or model is needed for validation'
        self.model = model
        model.eval()
        self.device = model.device  # update device
        imgsz = 640 
        if self.device.type == 'cpu':
            self.args.workers = 0  
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
        model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup
        n_batches = len(self.dataloader)
        bar = tqdm(self.dataloader, desc="Validation", total=n_batches, bar_format=TQDM_BAR_FORMAT,  
                   dynamic_ncols=True, leave=False)
        self.names = classes
        self.init_metrics(model)
        self.json_results = []  # empty before each val

        for batch_idx, batch in enumerate(bar):
            self.batch_idx = batch_idx
            batch = self.preprocess(batch)
            preds = model(batch['img']) 
            targets = get_targets(batch, self.device.type, self.args.task)
            loss_items = self.loss_function(preds, targets)[1]
            if not hasattr(self, 'loss') or self.loss.numel() == 0:
                self.loss = torch.zeros_like(loss_items, device=self.device.type)
            self.loss += loss_items
            if self.args.task == "detect" or self.args.task == "pose":
                preds = self.postprocess(preds)
            if self.args.task == "segment":
                preds, proto = self.postprocess(preds)

                if len(preds) == 0 or preds[0].shape[0] == 0:
                    #print("Skipped potting empty predictions!")
                    self.update_metrics(preds, batch) 
                    continue  # skip empty preds safely

            self.update_metrics(preds, batch)
            if self.args.plots and batch_idx < 3:
                self.plot_val_samples(batch, batch_idx)
                self.plot_predictions(batch, preds, batch_idx)
        print(self.metrics_summary_header())
        self.loss = self.loss / len(self.dataloader)
        stats = self.compute_metrics_results()
        self.finalize_metrics()
        self.print_results()
        self.dir_to_save()
        if self.args.save_json and self.json_results:
            with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                LOGGER.info(f'Saving {f.name}...')
                json.dump(self.json_results, f)  # flatten and save
            stats = self.eval_json(stats)  # update stats
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {self.save_dir}")
        return stats, self.loss

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError('get_dataloader function not implemented for this validator')

    def build_dataset(self, img_path):
        """Build dataset"""
        raise NotImplementedError('build_dataset function not implemented in validator')

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        return batch

    def postprocess(self, preds):
        """Describes and summarizes the purpose of 'postprocess()' but no details mentioned."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        pass

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes and returns all metrics."""
        pass

    def compute_metrics_results(self):
        """Returns statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Checks statistics."""
        pass

    def print_results(self):
        """Prints the results of the model's predictions."""
        pass

    def metrics_summary_header(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Returns the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        self.plots[name] = {'data': data, 'timestamp': time.time()}

    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plots YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass
