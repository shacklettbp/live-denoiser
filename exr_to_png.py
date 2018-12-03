import sys
import torch
import torchvision
from data_loading import load_exr, save_exr
from utils import tonemap

src = sys.argv[1]
dst = sys.argv[2]

exr = load_exr(src)
tonemapped = tonemap(exr)
clamped = torch.clamp(tonemapped, 0, 1.0)

txfm = torchvision.transforms.ToPILImage()
img = txfm(clamped)
img.save(dst)
