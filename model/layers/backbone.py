import torch.nn as nn
from .basic_modules import Conv, C2f, SPPF
from utils.models import make_n

class Backbone(nn.Module):
    def __init__(self, ch=3, base_channels=64, depth_multiple=1.0): #width_multiple=1.0,
        super().__init__()
        
        # Stem
        self.stem = Conv(ch, base_channels, 3, 2)  # Downsample from 640x640 -> 320x320

        # Stage 1
        self.stage1_conv = Conv(base_channels, base_channels * 2, 3, 2)  # 320 -> 160
        self.stage1_c2f = C2f(base_channels * 2, base_channels * 2, n=make_n(n = 3, depth_multiple = depth_multiple))

        # Stage 2
        self.stage2_conv = Conv(base_channels * 2, base_channels * 4, 3, 2)  # 160 -> 80
        self.stage2_c2f = C2f(base_channels * 4, base_channels * 4, n=make_n(n = 6, depth_multiple = depth_multiple))

        # Stage 3
        self.stage3_conv = Conv(base_channels * 4, base_channels * 8, 3, 2)  # 80 -> 40
        self.stage3_c2f = C2f(base_channels * 8, base_channels * 8, n=make_n(n = 6, depth_multiple = depth_multiple))

        # Stage 4
        self.stage4_conv = Conv(base_channels * 8, base_channels * 16, 3, 2)  # 40 -> 20
        self.stage4_c2f = C2f(base_channels * 16, base_channels * 16, n=make_n(n = 3, depth_multiple = depth_multiple))

        # SPPF
        self.sppf = SPPF(base_channels * 16, base_channels * 16)


    def forward(self, x):
        outputs = []

        x = self.stem(x)
        x = self.stage1_conv(x)
        x = self.stage1_c2f(x)
        outputs.append(x)  # P3 (160x160)
        
        x = self.stage2_conv(x)
        x = self.stage2_c2f(x)
        outputs.append(x)  # P4 (80x80)

        x = self.stage3_conv(x)
        x = self.stage3_c2f(x)
        outputs.append(x)  # P5 (40x40)

        x = self.stage4_conv(x)
        x = self.stage4_c2f(x)
        x = self.sppf(x)
        outputs.append(x)  # P6 (20x20)

        return outputs  # for neck: [P3, P4, P5, P6]
