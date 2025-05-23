import json
import os
import torch
import torch.nn as nn
from .layers.backbone import Backbone
from .layers.neck import Neck
from .layers.head import Detect, Segment, Pose
from utils.models import adjust_channels


current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "scale_config.json")

with open(config_path, "r") as f:
    yolov8_configs = json.load(f)


# YOLOv8 Model Class
class YOLOv8(nn.Module):
    """
    YOLOv8 model class that composes the backbone, neck, and head modules.

    Args:
        nc (int, optional): Number of classes. Defaults to 80.
        names_dict (dict, optional): Dictionary mapping class indices to names. Defaults to None.
        args (Namespace, optional): Additional training or configuration arguments. Defaults to None.
        backbone (nn.Module): The backbone module for feature extraction.
        neck (nn.Module): The neck module for feature fusion.
        head (nn.Module): The head module for prediction.

    Attributes:
        model (nn.ModuleList): Combined list of all model parts for unified access.
        device (torch.device): The device on which the model is loaded.
        stride (torch.Tensor): Computed stride values based on a dummy forward pass.
        names (dict): Class names dictionary.
        args (Namespace): Training/config arguments.
    """
    def __init__(
        self, nc=80, names_dict=None, args=None, backbone=None, neck=None, head=None
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        # Combine for loss compatibility
        self.model = nn.ModuleList([self.backbone, self.neck, self.head])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Build strides
        m = self.model[-1]
        s = 640  # 2x min stride
        ch = 3
        m.inplace = True
        forward = lambda x: (
            self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
        )
        m.stride = torch.tensor(
            [s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))]
        )  # forward
        self.stride = m.stride
        m.bias_init()  # only run once
        self.names = names_dict
        self.args = args

    def forward(self, x):
        """
        Forward pass through the YOLOv8 model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor or tuple: Output predictions from the model head.
        """
        # Backbone
        features = self.backbone(x)
        # Neck (FPN)
        fused_features = self.neck(features)
        # Head
        outputs = self.head(fused_features)
        return outputs

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warm up the model by running one forward pass with a dummy input.
        """
        # In case the model will be not on CPU (can't test now)
        if self.device.type != "cpu":
            im = torch.empty(*imgsz, dtype=torch.float, device=self.device)  # input
            self.forward(im)  # warmup
            

def download_model(model_path, model):
    """
    Loads model weights from file.

    Supports loading from either 'ema_state_dict', 'model_state_dict', or direct state_dict format.

    Args:
        model_path (str): Path to the model file.
        model (nn.Module): The YOLOv8 model instance to load weights into.

    Raises:
        KeyError: If file does not contain a valid state dict.
        TypeError: If file is not a dictionary.
    """
    print(f"Loading model from {model_path}...")
    ckpt = torch.load(model_path, map_location = model.device )
    if isinstance(ckpt, dict):
        if "ema_state_dict" in ckpt:
            print("Loading ema_state_dict...")
            model.load_state_dict(ckpt["ema_state_dict"])
        elif "model_state_dict" in ckpt:
            print("Loading model_state_dict...")
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            try:
                print("Trying to load checkpoint directly as state_dict...")
                model.load_state_dict(ckpt)
            except RuntimeError as e:
                raise KeyError(
                    "Checkpoint does not contain 'ema_state_dict' or 'model_state_dict', "
                    "and loading entire checkpoint as state_dict failed."
                ) from e
    else:
        raise TypeError(f"Unexpected checkpoint format: expected dict but got {type(ckpt)}")

# Function to create different YOLOv8 variants
def build_yolov8(
    model_size="n",
    num_classes=80,
    names_dict=None,
    task="detect",
    model_path=None,
    args=None
):
    """
    Builds a YOLOv8 model based on size and task specification.

    Constructs the backbone, neck, and task-specific head, then optionally loads pretrained weights.

    Args:
        model_size (str, optional): Size variant of the model ('n', 's', 'm', 'l', 'x'). Defaults to "n".
        num_classes (int, optional): Number of classes for the detection/segmentation/pose task. Defaults to 80.
        names_dict (dict, optional): Mapping from class indices to names. Defaults to None.
        task (str, optional): Task type - 'detect', 'segment', or 'pose'. Defaults to "detect".
        model_path (str, optional): Optional path to file to load pretrained weights. Defaults to None.
        args (Namespace, optional): Additional arguments. Defaults to None.

    Returns:
        YOLOv8: The constructed YOLOv8 model.
    """
    config = yolov8_configs[model_size]
    width_multiple = config["width_multiple"]
    depth_multiple = config["depth_multiple"]

    base_channels = 64

    backbone = Backbone(
        ch=3,
        base_channels=adjust_channels(base_channels, width_multiple),
        depth_multiple=depth_multiple,
    )
    neck = Neck(
        base_channels=adjust_channels(base_channels, width_multiple),
        depth_multiple=depth_multiple,
    )
    head_ch = [
        adjust_channels(base_channels * 2, width_multiple),
        adjust_channels(base_channels * 4, width_multiple),
        adjust_channels(base_channels * 8, width_multiple),
    ]
    if task == "detect":
        head = Detect(nc=num_classes, ch=head_ch)
    elif task == "segment":
        head = Segment(nc=num_classes, ch=head_ch)
    elif task == "pose":
        head = Pose(nc=num_classes, ch=head_ch)
    model = YOLOv8(
            nc=num_classes,
            names_dict=names_dict,
            backbone=backbone,
            neck=neck,
            head=head,
            args=args,
        )
    # If model_path is provided, load the weights
    if model_path:
        download_model(model_path, model)
    # Print parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    return model
