import torch
import torchvision
import OpenEXR
import Imath
import numpy as np
import torch.nn.functional as F
from utils import tonemap

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

def load_exr(filename):
    floattype = Imath.PixelType(Imath.PixelType.HALF)

    file = OpenEXR.InputFile(filename)
    dw = file.header()['dataWindow']

    channels_list = ['R', 'G', 'B']
    rgb_channels = file.channels(channels_list, floattype)
    decoded_channels = [np.fromstring(channel, dtype=np.float16) for channel in rgb_channels]

    imgsz = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    array = np.stack(decoded_channels, axis=0).reshape(3, *imgsz)

    tensor = torch.tensor(array).float()

    tensor[torch.isnan(tensor)] = 0
    tensor[torch.isinf(tensor)] = 0

    return tensor

def load_raw_crop(filename, offset_x, offset_y, cropsize, fullshape):
    mmapped = np.memmap(filename, dtype=np.float16, mode='r', shape=fullshape)
    crop = mmapped[:, offset_y:offset_y+cropsize[0], offset_x:offset_x+cropsize[1]]

    # Convert to full precision Tensor for now
    return torch.tensor(crop).float()

def load_raw(filename, fullshape):
    arr = np.fromfile(filename, dtype=np.float16).reshape(fullshape)
    return torch.tensor(arr).float()

def dump_raw(tensor, raw_filename):
    tensor.numpy().astype(np.float16).tofile(raw_filename)

def save_exr(tensor, filename):
    R = tensor[0, ...].cpu().numpy().astype(np.float16).tostring()
    G = tensor[1, ...].cpu().numpy().astype(np.float16).tostring()
    B = tensor[2, ...].cpu().numpy().astype(np.float16).tostring()
    header = OpenEXR.Header(tensor.shape[-1], tensor.shape[-2])
    channeltype = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    header['channels'] = { 'R': channeltype,
                           'G': channeltype,
                           'B': channeltype }
    header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
    out = OpenEXR.OutputFile(filename, header)
    out.writePixels({'R' : R, 'G': G, 'B': B })

def save_png(tensor, filename):
    img = torchvision.transforms.ToPILImage()(tonemap(tensor).clamp(0, 1.0).cpu())
    img.save(filename)
