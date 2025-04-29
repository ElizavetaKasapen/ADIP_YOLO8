from dataclasses import dataclass
from typing import List, Optional

@dataclass
class YoloDataConfig:
    path: str
    train: str
    val: str
    labels: str
    nc: int
    names: List[str]
    kpt_shape: Optional[List[int]] = None
    flip_idx: Optional[List[int]] = None
