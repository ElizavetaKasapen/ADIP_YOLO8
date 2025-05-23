from utils.io import LOGGER 
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter

# TypeError for handling 'Descriptors cannot not be created directly.' protobuf errors in Windows
except (ImportError, AssertionError, TypeError):
    SummaryWriter = None

writer = None  # TensorBoard SummaryWriter instance


def _log_scalars(scalars, step=0):
    """Logs scalar values to TensorBoard."""
    if scalars is None:
        print("TensorBoard: Expected a dictionary of scalars, got None")
        return  
    if writer:
        for k, v in scalars.items():
            writer.add_scalar(k, v, step)


def on_pretrain_routine_start(trainer):
    """Initialize TensorBoard logging with SummaryWriter."""
    if SummaryWriter:
        try:
            global writer
            run_name = f"{trainer.task}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            log_dir = Path(trainer.train_args.save_dir) / Path(run_name)
            writer = SummaryWriter(str(log_dir))
            prefix = 'TensorBoard: '
            LOGGER.info(f"{prefix}Start with 'tensorboard --logdir {log_dir}', view at http://localhost:6006/")
        except Exception as e:
            LOGGER.warning(f'WARNING TensorBoard not initialized correctly, not logging this run. {e}')


def on_fit_epoch_end(trainer):
    """Logs epoch metrics at end of training epoch."""
    _log_scalars(trainer.label_loss_items(trainer.loss_items, prefix='train'), trainer.epoch + 1)
    _log_scalars(trainer.metrics, trainer.epoch + 1)
    _log_scalars(trainer.label_loss_items(trainer.val_loss, prefix='val'), trainer.epoch + 1)


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_fit_epoch_end': on_fit_epoch_end}


def get_callbacks():
    """
    Return a copy of the default_callbacks dictionary with lists as default values.

    Returns:
        (defaultdict): A defaultdict with keys from default_callbacks and empty lists as default values.
    """
    return defaultdict(list, deepcopy(callbacks))