import torch 
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from PIL import Image
from utils.config import IMG_FORMATS
from .logger import LOGGER


@dataclass
class SourceTypes:
    from_img: bool = False
    tensor: bool = False

class LoadPilAndNumpy:

    def __init__(self, im0, imgsz=640):
        """Initialize PIL and Numpy Dataloader."""
        if not isinstance(im0, list):
            im0 = [im0]
        self.paths = [getattr(im, 'filename', f'image{i}.jpg') for i, im in enumerate(im0)]
        self.im0 = [self._single_check(im) for im in im0]
        self.imgsz = imgsz
        self.mode = 'image'
        # Generate fake paths
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im):
        """Validate and format an image to numpy array."""
        assert isinstance(im, (Image.Image, np.ndarray)), f'Expected PIL/np.ndarray image type, but got {type(im)}'
        if isinstance(im, Image.Image):
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def __len__(self):
        """Returns the length of the 'im0' attribute."""
        return len(self.im0)

    def __next__(self):
        """Returns batch paths, images, processed images, None, ''."""
        if self.count == 1:  # loop only once as it's batch inference
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, None, ''

    def __iter__(self):
        """Enables iteration for class LoadPilAndNumpy."""
        self.count = 0
        return self



class LoadTensor:

    def __init__(self, im0) -> None:
        self.im0 = self._single_check(im0)
        self.bs = self.im0.shape[0]
        self.mode = 'image'
        self.paths = [getattr(im, 'filename', f'image{i}.jpg') for i, im in enumerate(im0)]

    @staticmethod
    def _single_check(im, stride=32):
        """Validate and format an image to torch.Tensor."""
        s = f'WARNING  torch.Tensor inputs should be BCHW i.e. shape(1, 3, 640, 640) ' \
            f'divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible.'
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            LOGGER.warning(s)
            im = im.unsqueeze(0)
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)
        if im.max() > 1.0:
            LOGGER.warning(f'WARNING  torch.Tensor inputs should be normalized 0.0-1.0 but max value is {im.max()}. '
                           f'Dividing input by 255.')
            im = im.float() / 255.0

        return im

    def __iter__(self):
        """Returns an iterator object."""
        self.count = 0
        return self

    def __next__(self):
        """Return next item in the iterator."""
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, None, ''

    def __len__(self):
        """Returns the batch size."""
        return self.bs



def autocast_list(source):
    """
    Converts a list of inputs (paths, PIL images, numpy arrays) into a list of images.

    Allowed types: str, Path, PIL.Image.Image, np.ndarray
    """
    files = []
    for im in source:
        if isinstance(im, (str, Path)):  # local image path
            files.append(Image.open(im))
        elif isinstance(im, (Image.Image, np.ndarray)):  # PIL or np image
            files.append(im)
        else:
            raise TypeError(f'Unsupported type in list: {type(im).__name__}')
    return files


def check_source(source):
    """Strictly validates allowed input types."""
    from_img = tensor = False

    if isinstance(source, (str, Path)):
        ext = Path(source).suffix.lower()[1:]
        if ext not in IMG_FORMATS:
            raise TypeError(f"Unsupported file extension: {ext}")
        source = [Image.open(source)]  # force it to list of images
        from_img = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        source = [source]
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError('Only local image paths, PIL.Image, np.ndarray, torch.Tensor, or list of those are allowed.')

    return source, False, False, from_img, False, tensor



def load_inference_source(source=None, imgsz=640, vid_stride=1):
    """
    Loads static image formats: file path, tensor, PIL image, numpy array, or list of these.

    Args:
        source (str | Path | Tensor | PIL.Image | np.ndarray | list): Input source.
        imgsz (int): Resize dimension.

    Returns:
        dataset (Dataset): Wrapped dataset for inference.
    """
    source, _, _, from_img, _, tensor = check_source(source)
    source_type = SourceTypes(from_img=from_img, tensor=tensor) #webcam=False, screenshot=False,

    if tensor:
        dataset = LoadTensor(source)
    elif from_img:
        dataset = LoadPilAndNumpy(source, imgsz=imgsz)
    else:
        raise TypeError("Only image files, PIL.Image, np.ndarray, torch.Tensor or list of those are allowed.")

    setattr(dataset, 'source_type', source_type)
    return dataset