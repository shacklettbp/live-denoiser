import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import tonemap

num_input_channels = 9
kernel_size = 3
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, relu=True, leaky=True, kernel_size=(3, 3), batchnorm=False):
        super(Conv, self).__init__()
        padding = (1 if kernel_size[0] == 3 else 0, 1 if kernel_size[1] == 3 else 0)
        cur = [nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels, kernel_size=kernel_size,
                         padding=padding,
                         stride=stride)]

        if relu:
            if leaky:
                cur.append(nn.LeakyReLU(negative_slope=0.1))
            else:
                cur.append(nn.ReLU())

        if relu and leaky:
            self.init_nonlinearity = 'leaky_relu'
        elif relu:
            self.init_nonlinearity = 'relu'
        else:
            self.init_nonlinearity = 'linear'

        if batchnorm:
            cur.append(nn.BatchNorm2d(out_channels))

        self.model = nn.Sequential(*cur)

    def forward(self, inputs):
        out = self.model(inputs)
        return out

    def initialize(self):
        nn.init.kaiming_normal_(self.model[0].weight.data, nonlinearity=self.init_nonlinearity)

def init_weights(m):
    if isinstance(m, Conv):
        m.initialize()

class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True):
        super(TripleConv, self).__init__()
        self.model = nn.Sequential(Conv(in_channels=in_channels,
                                        out_channels=out_channels,
                                        relu=True, leaky=False),
                                   Conv(in_channels=out_channels,
                                        out_channels=out_channels,
                                        relu=True, leaky=False),
                                   Conv(in_channels=out_channels,
                                        out_channels=out_channels,
                                        relu=relu, leaky=False))

    def forward(self, inputs):
        return self.model(inputs)

class DenoiserModel(nn.Module):
    def __init__(self, init):
        super(DenoiserModel, self).__init__()
        self.enc_block1 = TripleConv(9, 32)
        self.enc_block2 = TripleConv(32, 64)
        self.enc_block3 = TripleConv(64, 96)
        self.enc_block4 = TripleConv(96, 128)
        self.enc_block5 = TripleConv(128, 256)

        self.bottleneck = TripleConv(256, 256)

        self.dec_block5 = TripleConv(256+256, 256)
        self.dec_block4 = TripleConv(256+128, 128)
        self.dec_block3 = TripleConv(128+96, 96)
        self.dec_block2 = TripleConv(96+64, 64)
        self.dec_block1 = TripleConv(64+32, 32)

        self.final = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1))

        if init:
            self.apply(init_weights)

    def forward(self, color, normal, albedo, color_prev1, color_prev2, albedo_prev1, albedo_prev2):
        eps = 0.001
        color = color / (albedo + eps)

        mapped_color = torch.log1p(color)
        mapped_albedo = torch.log1p(albedo)

        full_input = torch.cat([mapped_color, normal, mapped_albedo], dim=1)

        enc1_out = self.enc_block1(full_input)
        out = F.interpolate(enc1_out, scale_factor=0.5, mode='nearest')

        enc2_out = self.enc_block2(out)
        out = F.interpolate(enc2_out, scale_factor=0.5, mode='nearest')

        enc3_out = self.enc_block3(out)
        out = F.interpolate(enc3_out, scale_factor=0.5, mode='nearest')

        enc4_out = self.enc_block4(out)
        out = F.interpolate(enc4_out, scale_factor=0.5, mode='nearest')

        enc5_out = self.enc_block5(out)
        out = F.interpolate(enc5_out, scale_factor=0.5, mode='nearest')

        out = self.bottleneck(out)
        out = F.interpolate(out, scale_factor=2, mode='nearest')

        out = torch.cat([out, enc5_out], dim=1)
        out = self.dec_block5(out)
        out = F.interpolate(out, scale_factor=2, mode='nearest')

        out = torch.cat([out, enc4_out], dim=1)
        out = self.dec_block4(out)
        out = F.interpolate(out, scale_factor=2, mode='nearest')

        out = torch.cat([out, enc3_out], dim=1)
        out = self.dec_block3(out)
        out = F.interpolate(out, scale_factor=2, mode='nearest')

        out = torch.cat([out, enc2_out], dim=1)
        out = self.dec_block2(out)
        out = F.interpolate(out, scale_factor=2, mode='nearest')

        out = torch.cat([out, enc1_out], dim=1)
        out = self.dec_block1(out)

        out = self.final(out)

        exp = torch.expm1(out)

        return exp * (albedo + eps), exp

class TemporalDenoiserModel(nn.Module):
    def __init__(self, recurrent, *args, **kwargs):
        super(TemporalDenoiserModel, self).__init__()
        self.recurrent = recurrent
        self.model = DenoiserModel(*args, **kwargs)

    def forward(self, color, normal, albedo, color_prev1=None, color_prev2=None, albedo_prev1=None, albedo_prev2=None):
        color = color.transpose(0, 1)
        normal = normal.transpose(0, 1)
        albedo = albedo.transpose(0, 1)

        if color_prev1 is None:
            color_prev1 = torch.zeros_like(color[0])
        if color_prev2 is None:
            color_prev2 = torch.zeros_like(color[0])

        if albedo_prev1 is None:
            albedo_prev1 = torch.zeros_like(albedo[0])
        if albedo_prev2 is None:
            albedo_prev2 = torch.zeros_like(albedo[0])

        all_outputs = []
        e_irradiances = []
        for i in range(color.shape[0]):
            output, e_irradiance = self.model(color[i], normal[i], albedo[i], color_prev1, color_prev2, albedo_prev1, albedo_prev2)
            color_prev2 = color_prev1
            albedo_prev2 = albedo_prev1

            if self.recurrent:
                color_prev1 = output
            else:
                color_prev1 = color[i]

            albedo_prev1 = albedo[i]
            
            all_outputs.append(output)
            e_irradiances.append(e_irradiance)

        return torch.stack(all_outputs, dim=1), torch.stack(e_irradiances, dim=1)
