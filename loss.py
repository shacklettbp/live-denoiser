import torch
import torch.nn.functional as F
from utils import calc_luminance

class Loss:
    def __init__(self, dev, config):
        self.gaussian = torch.tensor([0.035822, 0.05879, 0.086425, 0.113806, 0.13424, 0.141836, 0.13424, 0.113806, 0.086425, 0.05879, 0.035822], device=dev)

        self.config = config

        configs = ["n2r_trad", "n2n_trad", "n2r_spatial_l2", "n2n_spatial_l2"]

        if self.config not in configs:
            print("Error: loss config '%s' not supported" % self.config)

        print("************ Initialized loss as %s ************" % self.config)

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

    def spatial_gradients_l2(self, true, pred):
        d_x_true = (true[..., 1:]    - true[..., 0:-1])
        d_y_true = (true[..., 1:, :] - true[..., 0:-1, :])
        d_x_pred = (pred[..., 1:]    - pred[..., 0:-1])
        d_y_pred = (pred[..., 1:, :] - pred[..., 0:-1, :])

        sgl2 = torch.mean(torch.pow(d_x_true - d_x_pred, 2)) + torch.mean(torch.pow(d_y_true - d_y_pred, 2))

        return sgl2

    def compute_l2(self, out, ref):
        if self.config.startswith("n2r_"):
            return (((out - ref)**2)/(ref.detach()**2 + 0.001)).mean() # N2R
        elif self.config.startswith("n2n_"):
            return (((out - ref)**2)/(out.detach()**2 + 0.001)).mean() # N2N
        else:
            print("Error: Invalid loss configuration (neither n2r nor n2n)")
            return None

    def compute(self, refs, outputs, ref_e_irradiance, e_irradiance, ref_albedos, albedos):
        assert(len(refs.shape) == 5 and len(outputs.shape) == 5) # We need the temporal component

        #noise2noise_loss = self.compute_l2(out, ref)
        noise2noise_loss = 0

        #lum_loss = self.compute_luminance_reg(out, input)
    
        #loss = noise2noise_loss.mean()# + 1e-4 * lum_loss.mean() + 1e-3 * (d_x.mean() + d_y.mean())
        # sgl2 = self.spatial_gradients_l2(outputs, refs)

        ref_temporal_gradients = refs[:, 0:-1, ...] - refs[:, 1:, ...]
        output_temporal_gradients = outputs[:, 0:-1, ...] - outputs[:, 1:, ...]

        irradiance_loss = self.compute_l2(e_irradiance, ref_e_irradiance)
        temporal_loss   = self.compute_l2(output_temporal_gradients, ref_temporal_gradients)
        albedo_loss     = self.compute_l2(albedos, ref_albedos)
        spatial_l2      = self.compute_l2(outputs, refs)

        if self.config.endswith("_trad"):
            loss = irradiance_loss + temporal_loss + albedo_loss
        elif self.config.endswith("_spatial_l2"):
            loss = temporal_loss + spatial_l2
        else:
            print("Error: Invalid loss configuration (neither trad nor spatial_l2)")
            loss = 0.0

        return loss, irradiance_loss, temporal_loss, albedo_loss
