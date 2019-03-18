import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def simple_filter(color, factor):
    small_height = color.shape[2] // factor
    small_width = color.shape[3] // factor

    small_color = torch.zeros(*color.shape[0:2], small_height, small_width, dtype=color.dtype, device=color.device)
    for y in range(0, small_height):
        for x in range(0, small_width):
            small_color[..., y, x] = color[..., y*factor:y*factor+factor, x*factor:x*factor+factor].mean(dim=[2, 3])

    blurred_color = F.interpolate(small_color, scale_factor=factor, mode='bilinear')

    return blurred_color

def bilateral_filter(color, normals):
    kernel_size = 65

    with torch.no_grad():
        width, height = color.shape[-1], color.shape[-2] 

        filtered = torch.zeros_like(color)
        weight_sum = torch.zeros_like(color[:, 0:1, ...])

        pad_amount = (kernel_size - 1) // 2
        center_pos = pad_amount

        color = F.pad(color, (pad_amount, pad_amount, pad_amount, pad_amount), mode='replicate')
        normals = F.pad(normals, (pad_amount, pad_amount, pad_amount, pad_amount), mode='replicate')

        for y in range(kernel_size):
            for x in range(kernel_size):
                sigma = 20
                gauss_weight = math.exp(-((y - center_pos)**2 + (x - center_pos)**2)/(2*sigma**2))/(2*math.pi*sigma**2)

                normal_diff = torch.abs(normals[:, :, y:y + height, x:x + width] - normals[:, :, center_pos:center_pos+height, center_pos:center_pos+width]).max(dim=1, keepdim=True)[0]
                normal_dist = (4.0*(-normal_diff + 0.25)).clamp(0, 1)

                shifted_color = color[:, :, y:y + height, x:x + width]
                weights = gauss_weight*normal_dist

                weight_sum = weight_sum + weights

                filtered = filtered + weights * shifted_color

        filtered = filtered / weight_sum

    return filtered
