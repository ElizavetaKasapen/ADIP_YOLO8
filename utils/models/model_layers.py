import math

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def make_divisible(x, divisor=8):
    """Make channel number divisible by divisor (default 8)"""
    return math.ceil(x / divisor) * divisor

def adjust_channels(channels, width_multiple):
    return make_divisible(channels * width_multiple, 8)

def make_n(n, depth_multiple):
    return max(round(n * depth_multiple), 1)
