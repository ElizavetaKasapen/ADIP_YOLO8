import torch
import torch.nn as nn
from .basic_modules import Conv, C2f
from utils.model.model_scales import adjust_channels, make_n

class Neck(nn.Module):
    def __init__(self, base_channels=64,  depth_multiple=1.0): #width_multiple = 1.0,
        super().__init__()

        def make_n(n): return max(round(n * depth_multiple), 1)
        # c2 = adjust_channels(base_channels * 2, width_multiple)   # 128
        # c3 = adjust_channels(base_channels * 4, width_multiple)   # 256
        # c4 = adjust_channels(base_channels * 8, width_multiple)   # 512
        # c5 = adjust_channels(base_channels * 16, width_multiple)  # 1024

        # Top-down
        self.reduce_layer1 = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f1 = C2f(base_channels * 8, base_channels * 4, n=make_n(3))  # 256 + 256 = 512

        self.reduce_layer2 = Conv(base_channels * 4, base_channels * 2, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f2 = C2f(base_channels * 4, base_channels * 2, n=make_n(3))  # 128 + 128 = 256

        # Bottom-up
        self.downsample1 = Conv(base_channels * 2, base_channels * 2, 3, 2)
        self.c2f3 = C2f(base_channels * 6, base_channels * 4, n=make_n(3))  # 128 + 256 = 384

        self.downsample2 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.c2f4 = C2f(base_channels * 12, base_channels * 8, n=make_n(3)) # 256 + 512 = 768



    def forward(self, inputs):
        [p3, p4, p5, _] = inputs  # Ignore P6

        # Top-down
        p5_reduced = self.reduce_layer1(p5)
        p5_upsampled = self.upsample1(p5_reduced)
        p4_td = self.c2f1(torch.cat([p5_upsampled, p4], dim=1))

        p4_reduced = self.reduce_layer2(p4_td)
        p4_upsampled = self.upsample2(p4_reduced)
        p3_out = self.c2f2(torch.cat([p4_upsampled, p3], dim=1))

        # Bottom-up
        p3_down = self.downsample1(p3_out)
        p4_out = self.c2f3(torch.cat([p3_down, p4_td], dim=1))

        p4_down = self.downsample2(p4_out)
        p5_out = self.c2f4(torch.cat([p4_down, p5], dim=1))

        return [p3_out, p4_out, p5_out]
