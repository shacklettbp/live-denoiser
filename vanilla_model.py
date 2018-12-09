import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import tonemap

num_input_channels = 9
kernel_size = 5

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
    def __init__(self, init, use_dfn=True):
        super(VanillaDenoiserModel, self).__init__()
        use_dfn = False
        self.use_dfn = use_dfn
        num_output_channels = kernel_size**2 if use_dfn else 3

        self.encoder = DenoiserEncoder([[48, 48], [48], [48], [48], [48], [48]],
                                       [num_input_channels, 48, 48, 48, 48, 48])

        self.decoder = DenoiserDecoder([[96, 96], [96, 96], [96, 96], [96, 96], [64, 32]],
                                       [48+48, 96+48, 96+48, 96+48, 96+num_input_channels],
                                       num_output_channels)

        if init:
            self.apply(init_weights)


    def forward(self, color, normal, albedo):
        tonemapped = tonemap(color)

        full_input = torch.cat([tonemapped, normal, albedo], dim=1)

        enc_outs = self.encoder(full_input)

        output = self.decoder(enc_outs)

        if self.use_dfn:
            return self.dynamic_filters(color, output)
        else:
            return output

    def dynamic_filters(self, color, weights):
        weights = weights.view(-1, kernel_size, kernel_size, weights.shape[-2], weights.shape[-1])
        width, height = color.shape[-1], color.shape[-2]
        output = torch.zeros_like(color)
        padding = kernel_size // 2
        padded = F.pad(color, (padding, padding, padding, padding))

        for i in range(kernel_size):
            for j in range(kernel_size):
                cur_weights = weights[:, i, j, ...].unsqueeze(dim=1)
                offset_y = i - kernel_size // 2
                offset_x = j - kernel_size // 2
                output = output + padded[:, :, padding + offset_y:padding + offset_y + height, padding + offset_x:padding + offset_x + width] * cur_weights

        return output

