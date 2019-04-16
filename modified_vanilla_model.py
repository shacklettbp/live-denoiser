import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import tonemap, ycocg, rgb
from filters import simple_filter

num_input_channels = 17
num_output_channels = 3
kernel_size = 3

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True, leaky=True):
        super(Conv, self).__init__()
        cur = [nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=3,
                         padding=1)]
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

        self.model = nn.Sequential(*cur)

    def forward(self, inputs):
        return self.model(inputs)

    def initialize(self):
        nn.init.kaiming_normal_(self.model[0].weight.data, nonlinearity=self.init_nonlinearity)

def create_layers(sizes_num_layers, layer_input_sizes):
    layers = nn.ModuleList()

    for idx, sizes in enumerate(sizes_num_layers):
        prev_size = layer_input_sizes[idx]
        cur = []
        for size in sizes:
            cur.append(Conv(in_channels=prev_size,
                       out_channels=size))
            prev_size=size

        layers.append(nn.Sequential(*cur))

    return layers

class DenoiserEncoder(nn.Module):
    def __init__(self, sizes_num_layers, layer_input_sizes):
        super(DenoiserEncoder, self).__init__()
        self.encoder_layers = create_layers(sizes_num_layers, layer_input_sizes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        cur = inputs
        outputs = [inputs]
        for i, enc in enumerate(self.encoder_layers):
            cur = enc(cur)
            if i != len(self.encoder_layers) - 1:
                cur = self.pool(cur)
            if i != len(self.encoder_layers) - 2: # Pool 5 isn't used by decoder
                outputs.append(cur)

        return outputs

class DenoiserDecoder(nn.Module):
    def __init__(self, sizes_num_layers, layer_input_sizes, num_output_channels):
        super(DenoiserDecoder, self).__init__()
        self.decoder_layers = create_layers(sizes_num_layers, layer_input_sizes)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.final = Conv(in_channels=sizes_num_layers[-1][-1],
                          out_channels=num_output_channels,
                          relu=False)

    def forward(self, enc_outputs):
        cur = enc_outputs[-1]
        for dec, enc_output in zip(self.decoder_layers, reversed(enc_outputs[:-1])):
            cur = self.upsample(cur)
            cur = torch.cat([cur, enc_output], dim=1)
            cur = dec(cur)

        return self.final(cur)

def init_weights(m):
    if isinstance(m, Conv):
        m.initialize()

class VanillaDenoiserModel(nn.Module):
    def __init__(self, init):
        super(VanillaDenoiserModel, self).__init__()

        self.encoder = DenoiserEncoder([[48, 48], [48], [48], [48], [48], [48]],
                                       [num_input_channels, 48, 48, 48, 48, 48])

        self.decoder = DenoiserDecoder([[96, 96], [96, 96], [96, 96], [96, 96], [64, 32]],
                                       [48+48, 96+48, 96+48, 96+48, 96+num_input_channels],
                                       num_output_channels)

        if init:
            self.apply(init_weights)

    def dynamic_filters(self, color, manually_denoised, weights):
        width, height = color.shape[-1], color.shape[-2]

        #dense_weights = F.softmax(weights[:, 0:31, ...], dim=1)
        dense_weights = F.softmax(weights[:, 0:kernel_size*kernel_size*2, ...], dim=1)
        padding = kernel_size // 2
        padded_color = F.pad(color, (padding, padding, padding, padding))
        padded_manually_denoised = F.pad(manually_denoised, (padding, padding, padding, padding))

        shifted = []
        for i in range(kernel_size):
            for j in range(kernel_size):
                shifted.append(padded_color[:, :, i:i + height, j:j + width])

        for i in range(kernel_size):
            for j in range(kernel_size):
                shifted.append(padded_manually_denoised[:, :, i:i + height, j:j + width])

        img_stack = torch.stack(shifted, dim=1)
        dense_output = torch.sum(dense_weights.unsqueeze(dim=2) * img_stack, dim=1).squeeze(dim=1)

        return dense_output

    def forward(self, color, normal, albedo, prev1, prev2):
        eps = 0.001
        color = color / (albedo + eps)

        prefiltered = simple_filter(color, factor=64)

        mapped_color = torch.log1p(color)
        mapped_albedo = torch.log1p(albedo)
        mapped_prefiltered = torch.log1p(prefiltered)
        mapped_prev1 = torch.log1p(prev1)
        mapped_prev2 = torch.log1p(prev2)

        #mapped_color = ycocg(mapped_color)
        #mapped_albedo = ycocg(mapped_albedo)

        full_input = torch.cat([mapped_color, normal, mapped_albedo, mapped_prefiltered, mapped_prev1, mapped_prev2], dim=1)

        enc_outs = self.encoder(full_input)

        output = self.decoder(enc_outs)

        #output = rgb(output)

        exp = torch.expm1(output)

        return exp * (albedo + eps), exp 

class TemporalVanillaDenoiserModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TemporalVanillaDenoiserModel, self).__init__()
        self.model = VanillaDenoiserModel(*args, **kwargs)

    def forward(self, color, normal, albedo):
        color = color.transpose(0, 1)
        normal = normal.transpose(0, 1)
        albedo = albedo.transpose(0, 1)

        all_outputs = []
        ei_outputs = []

        prev1 = torch.zeros_like(color[0]);
        prev2 = torch.zeros_like(prev1);

        for i in range(color.shape[0]):
            output, ei = self.model(color[i], normal[i], albedo[i], prev1, prev2)
            all_outputs.append(output)
            ei_outputs.append(ei)

            prev2 = prev1
            prev1 = ei

        return torch.stack(all_outputs, dim=1), torch.stack(ei_outputs, dim=1)

class VanillaDenoiserModelWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super(VanillaDenoiserModelWrapper, self).__init__()
        self.model = VanillaDenoiserModel(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
