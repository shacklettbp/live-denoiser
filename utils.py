import torch
import torch.nn.functional as F

def roundup(num, mul):
    rounded = (num // mul)
    if num % mul != 0:
        rounded += 1

    return rounded * mul

def pad_data(tensor, mul=32):
    height, width = tensor.shape[-2:]

    rounded_height = roundup(height, mul)
    rounded_width = roundup(width, mul)

    height_pad = rounded_height - height
    width_pad = rounded_width - width
    pad = (0, width_pad, 0, height_pad)

    return F.pad(tensor, pad, 'constant', 0)

def ycocg(tensor):
    return tensor
    r = tensor[:, 0, ...]
    g = tensor[:, 1, ...]
    b = tensor[:, 2, ...]

    y = r / 4 + g / 2 + b / 4
    co = r / 2 - b / 2
    cg = -r / 4 + g / 2 - b / 4

    return torch.stack([y, co, cg], dim=1)

def rgb(tensor):
    return tensor

    y = tensor[:, 0, ...]
    co = tensor[:, 1, ...]
    cg = tensor[:, 2, ...]

    r = y + co - cg
    g = y + cg
    b = y - co - cg

    return torch.stack([r, g, b], dim=1)

def calc_luminance(color):
    if len(color.shape) == 4:
        r = color[:, 0:1, ...]
        g = color[:, 1:2, ...]
        b = color[:, 2:3, ...]
    elif len(color.shape) == 3:
        r = color[0:1, ...]
        g = color[1:2, ...]
        b = color[2:3, ...]

    return r / 4 + g/2 + b/4

def tonemap(color, exposure_key=16):
    y = calc_luminance(color)

    y_exposed = exposure_key*y

    y_tm = y_exposed/(1 + y_exposed)
    y_denom = y
    scale = y_tm / (y_denom + 1e-12)

    return (color*scale)**(1/2.2)

# Hack because dataloader has no way of setting preferred device
def iter_with_device(dataloader, dev):
    with torch.cuda.device(dev):
        it = iter(dataloader)
    while True:
        with torch.cuda.device(dev):
            try:
                yield next(it)
            except StopIteration:
                return None
