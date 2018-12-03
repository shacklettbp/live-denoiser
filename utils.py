import torch

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

def tonemap(color):
    y = calc_luminance(color)

    y_exposed = 4*y

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
