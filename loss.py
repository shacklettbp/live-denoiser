import torch
import torch.nn.functional as F
from utils import calc_luminance

class Loss:
    def __init__(self, dev):
        self.gaussian = torch.tensor([0.035822, 0.05879, 0.086425, 0.113806, 0.13424, 0.141836, 0.13424, 0.113806, 0.086425, 0.05879, 0.035822], device=dev)

    def smooth(self, input):
        padded = F.pad(input, (5, 5, 5, 5), 'replicate')
        vert = F.conv2d(padded, self.gaussian.view(1, 1, 11, 1).expand(input.shape[-3], -1, -1, -1), groups=input.shape[-3])
        horz = F.conv2d(vert, self.gaussian.view(1, 1, 1, 11).expand(input.shape[-3], -1, -1, -1), groups=input.shape[-3])

        return horz

    def compute_luminance_reg(self, out, input):
        orig_shape = out.shape

        out = out.view(-1, *orig_shape[2:])
        input = input.view(out.shape)

        out_y = calc_luminance(out)
        input_y = calc_luminance(input)
        smoothed_input = self.smooth(input_y)
 
        return ((out_y - smoothed_input)**2).view(*orig_shape[:2], 1, *orig_shape[3:])

    def compute_spatial_reg(self, e_irradiance):
        height, width = e_irradiance.shape[-2:]
        d_x = (e_irradiance[..., 0:width-1] - e_irradiance[..., 1:width])**2
        d_y = (e_irradiance[..., 0:height-1, :] - e_irradiance[..., 1:height, :])**2

        return d_x, d_y
    
    def compute(self, refs, outputs, ref_e_irradiance, e_irradiance, ref_albedos, albedos):
        assert(len(refs.shape) == 5 and len(outputs.shape) == 5) # We need the temporal component

        #noise2noise_loss = (out - ref)**2/(out.detach()**2 + 0.01)
        noise2noise_loss = 0

        #lum_loss = self.compute_luminance_reg(out, input)

        #d_x, d_y = self.compute_spatial_reg(e_irradiance)
    
        #loss = noise2noise_loss.mean()# + 1e-4 * lum_loss.mean() + 1e-3 * (d_x.mean() + d_y.mean())
        irradiance_loss = (((e_irradiance - ref_e_irradiance)**2)/(e_irradiance.detach()**2 + 0.01)).mean()

        ref_temporal_gradients = refs[:, 0:2, ...] - refs[:, 1:3, ...]
        output_temporal_gradients = outputs[:, 0:2, ...] - outputs[:, 1:3, ...]

        temporal_loss = (((output_temporal_gradients - ref_temporal_gradients)**2)/(output_temporal_gradients.detach()**2 + 0.01)).mean()

        albedo_loss = (((albedos - ref_albedos)**2)/(albedos.detach()**2 + 0.01)).mean()
        #albedo_loss = ((albedos - ref_albedos)**2).mean()
        
        loss = irradiance_loss + temporal_loss + albedo_loss

        return loss, irradiance_loss, temporal_loss, albedo_loss
