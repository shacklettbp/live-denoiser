import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import tonemap

num_input_channels = 9
kernel_size = 3
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, relu=True, leaky=True, kernel_size=(3, 3), batchnorm=True):
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

class JitNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, upsample=1):
        super(JitNetBlock, self).__init__()
        self.main = nn.Sequential(Conv(in_channels=in_channels,
                                       out_channels=out_channels,
                                       stride=stride),
                                  Conv(in_channels=out_channels,
                                       out_channels=out_channels, kernel_size=(1,3),
                                       relu=False,
                                       batchnorm=False),
                                  Conv(in_channels=out_channels,
                                       out_channels=out_channels,
                                       kernel_size=(3,1),
                                       relu=False,
                                       batchnorm=False))

        if in_channels != out_channels:
            self.skip = Conv(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=(1,1),
                             stride=stride,
                             relu=False,
                             batchnorm=False)
        elif stride > 1:
            self.skip = nn.MaxPool2d(kernel_size=stride, stride=stride)
        else:
            self.skip = None

        self.post = nn.Sequential(nn.ReLU(),
                                  nn.BatchNorm2d(out_channels))

        self.upsample = upsample

    def forward(self, input):
        out = self.main(input)
        if self.skip is not None:
            out = out + self.skip(input)

        out = self.post(out)
        if self.upsample > 1:
            out = F.interpolate(out, scale_factor=self.upsample, mode='nearest')

        return out

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
        outputs = []
        for i, enc in enumerate(self.encoder_layers):
            cur = enc(cur)
            outputs.append(cur)
            if i != len(self.encoder_layers) - 1:
                cur = self.pool(cur)

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

def compute_luminance(color):
    return color[:, 0, ...] * 0.299 + color[:, 1, ...] * 0.587 + color[:, 2, ...]  * 0.114

def compute_exposure(img):
    log_mean = torch.log1p(compute_luminance(img)).view(img.shape[0], -1).mean(dim=1)
    avg_luminance = torch.expm1(log_mean).view(img.shape[0], 1, 1, 1)

    key = 1.03 - 2/(torch.log1p(avg_luminance) + 2)
    #key = 0.00001
    return key / (avg_luminance + 1e-5)

class DenoiserModel(nn.Module):
    def __init__(self, init):
        super(DenoiserModel, self).__init__()
        self.filter_func = self.direct_prediction
        #self.filter_func = self.albedo_prediction
        #self.kernel_size = 21
        #self.filter_func = self.mitchell_netravali

        #self.encoder = DenoiserEncoder([[48, 48], [48], [48], [48], [48], [48]],
        #                               [num_input_channels, 48, 48, 48, 48, 48])

        #self.decoder = DenoiserDecoder([[96, 96], [96, 96], [96, 96], [96, 96], [64, 32]],
        #                               [48+48, 96+48, 96+48, 96+48, 96+48],
        #                               num_output_channels)

        self.mixstart = nn.Sequential(nn.Conv2d(in_channels=num_input_channels,
                                                out_channels=16,
                                                stride=1,
                                                kernel_size=(1, 1)),
                                      nn.ReLU(),
                                      nn.BatchNorm2d(16),
                                      nn.Conv2d(in_channels=16,
                                                out_channels=16,
                                                groups=16,
                                                kernel_size=(3, 3),
                                                padding=(1, 1)),
                                      nn.Conv2d(in_channels=16,
                                                out_channels=16,
                                                kernel_size=(1, 1)),
                                      nn.ReLU(),
                                      nn.BatchNorm2d(16))

        self.enc_block1 = JitNetBlock(in_channels=16,
                                      out_channels=16,
                                      stride=2)

        self.enc_block2 = JitNetBlock(in_channels=16,
                                      out_channels=32,
                                      stride=2)

        self.enc_block3 = JitNetBlock(in_channels=32,
                                      out_channels=64,
                                      stride=2)
        self.enc_block4 = JitNetBlock(in_channels=64,
                                      out_channels=64,
                                      stride=2)

        self.enc_block5 = JitNetBlock(in_channels=64,
                                      out_channels=128,
                                      stride=2)

        self.dec_block5 = JitNetBlock(in_channels=128,
                                      out_channels=128,
                                      upsample=2)

        self.dec_block4 = JitNetBlock(in_channels=128,
                                      out_channels=64,
                                      upsample=2)

        self.dec_block3 = JitNetBlock(in_channels=64,
                                      out_channels=64,
                                      upsample=2)

        self.dec_block2 = JitNetBlock(in_channels=64,
                                      out_channels=32,
                                      upsample=2)

        self.dec_block1 = JitNetBlock(in_channels=32,
                                      out_channels=32,
                                      upsample=2)

        self.final = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32,
                                             kernel_size=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=32, out_channels=32,
                                             kernel_size=(3, 3),
                                             groups=32,
                                             padding=(1, 1)),
                                   nn.Conv2d(in_channels=32,
                                             out_channels=3,
                                             kernel_size=(1, 1),
                                             padding=0))

        if init:
            self.apply(init_weights)

    def forward(self, color, normal, albedo, color_prev1, color_prev2, albedo_prev1, albedo_prev2):
        eps = 0.001
        color = color / (albedo + eps)
        #color_prev1 = color_prev1 / (albedo_prev1 + eps)
        #color_prev2 = color_prev2 / (albedo_prev2 + eps)

        assert(not torch.isnan(color).any() and not (color == float('inf')).any())
        assert(not torch.isnan(normal).any() and not (normal == float('inf')).any())
        assert(not torch.isnan(albedo).any() and not (albedo == float('inf')).any())
        #assert(not torch.isnan(color_prev1).any() and not (color_prev1 == float('inf')).any())
        #assert(not torch.isnan(color_prev2).any() and not (color_prev2 == float('inf')).any())

        mapped_color = torch.log1p(color)
        mapped_normal = normal
        mapped_albedo = torch.log1p(albedo)
        #mapped_color_prev1 = torch.log1p(color_prev1.clamp(min=-0.9))
        #mapped_color_prev2 = torch.log1p(color_prev2.clamp(min=-0.9))

        #full_input = torch.cat([mapped_color, mapped_normal, mapped_albedo, mapped_color_prev1, mapped_color_prev2], dim=1)
        full_input = torch.cat([mapped_color, mapped_normal, mapped_albedo], dim=1)

        mixed = self.mixstart(full_input)
        enc1_out = self.enc_block1(mixed)
        enc2_out = self.enc_block2(enc1_out)
        enc3_out = self.enc_block3(enc2_out)
        enc4_out = self.enc_block4(enc3_out)
        enc5_out = self.enc_block5(enc4_out)
        out = self.dec_block5(enc5_out)
        out = torch.cat([out[:, 0:64, ...] + enc4_out, out[:, 64:, ...]], dim=1)
        out = self.dec_block4(out)
        out = out + enc3_out
        out = self.dec_block3(out)
        out = torch.cat([out[:, 0:32, ...] + enc2_out, out[:, 32:, ...]], dim=1)
        out = self.dec_block2(out)
        out = torch.cat([out[:, 0:16, ...] + enc1_out, out[:, 16:, ...]], dim=1)
        out = self.dec_block1(out)

        out = torch.cat([out[:, 0:16, ...] + mixed, out[:, 16:, ...]], dim=1)
        output = self.final(out)

        return self.filter_func(color, albedo, output) * (albedo + eps)

    def albedo_prediction(self, color, albedo, output):
        return albedo * torch.expm1(output)

    def direct_prediction(self, color, albedo, output):
        return torch.expm1(output)
        #return output

    def make_mn_weights(self, params):
        width, height = params.shape[-1], params.shape[-2]
        kern_size = 101

        radius = self.kernel_size // 2

        weights = params.new(params.shape[0], self.kernel_size, height, width).zero_()
        inv_rad = 1 / radius

        C = params[:, 1, ...]
        B = params[:, 0, ...]
        #B = 1 - 2*C

        mid_idx = radius

        # Make weights
        for i in range(0, radius + 1):
            x = 2*i*inv_rad
            if x > 1:
                tmp = ((-B - 6*C) * x*x*x + (6*B + 30*C) * x*x +
                       (-12*B - 48*C) * x + (8*B + 24*C)) * (1/6)
            else:
                tmp = ((12 - 9*B - 6*C) * x*x*x +
                       (-18 + 12*B + 6*C) * x*x +
                       (6 - 2*B)) * (1/6)

            weights[:, mid_idx + i, ...] = tmp
            weights[:, mid_idx - i, ...] = tmp

        return weights

    def mitchell_netravali(self, color, albedo, params):
        width, height = color.shape[-1], color.shape[-2]

        radius = self.kernel_size // 2

        padded = F.pad(color, (0, 0, radius, radius))
        filtered = torch.zeros_like(color)

        combo_weights = F.softmax(params[:, 0:3, ...], dim=1)

        weights = combo_weights[:, 0:1, ...] * self.make_mn_weights(params[:, 3:5, ...]) + combo_weights[:, 1:2, ...] * self.make_mn_weights(params[:, 5:7, ...]) + combo_weights[:, 2:3, ...] * self.make_mn_weights(params[:, 7:9, ...])

        shifted = []
        for i in range(self.kernel_size):
            shifted.append(padded[:, :, i:i+height, :])

        img_stack = torch.stack(shifted, dim=1)
        out_y = torch.sum(weights.unsqueeze(dim=2) * img_stack, dim=1).squeeze(dim=1)
        out_y = F.pad(out_y, (radius, radius, 0, 0))

        shifted = []
        for i in range(self.kernel_size):
            shifted.append(out_y[:, :, :, i:i+width])

        img_stack = torch.stack(shifted, dim=1)
        out = torch.sum(weights.unsqueeze(dim=2) * img_stack, dim=1).squeeze(dim=1)

        return out

    def dynamic_filters(self, color, weights):
        width, height = color.shape[-1], color.shape[-2]

        separable_x_weights = F.softmax(weights[:, 9:20, ...], dim=1)
        separable_y_weights = F.softmax(weights[:, 20:31, ...], dim=1)
        separable_padding = 11 // 2
        padded = F.pad(color, (0, 0, separable_padding, separable_padding))
        shifted = []
        for i in range(11):
            shifted.append(padded[:, :, i:i+height, :])

        img_stack = torch.stack(shifted, dim=1)
        out_separable_x = torch.sum(separable_x_weights.unsqueeze(dim=2) * img_stack, dim=1).squeeze(dim=1)

        padded = F.pad(out_separable_x, (separable_padding, separable_padding, 0, 0))
        shifted = []
        for i in range(11):
            shifted.append(padded[:, :, :, i:i+width])

        img_stack = torch.stack(shifted, dim=1)
        separable_output = torch.sum(separable_y_weights.unsqueeze(dim=2) * img_stack, dim=1).squeeze(dim=1)

        dense_weights = F.softmax(weights[:, 0:9, ...], dim=1)
        padding = kernel_size // 2
        padded = F.pad(separable_output, (padding, padding, padding, padding))

        shifted = []
        for i in range(kernel_size):
            for j in range(kernel_size):
                shifted.append(padded[:, :, i:i + height, j:j + width])

        img_stack = torch.stack(shifted, dim=1)
        dense_output = torch.sum(dense_weights.unsqueeze(dim=2) * img_stack, dim=1).squeeze(dim=1)

        return dense_output

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
        for i in range(color.shape[0]):
            output = self.model(color[i], normal[i], albedo[i], color_prev1, color_prev2, albedo_prev1, albedo_prev2)
            color_prev2 = color_prev1
            albedo_prev2 = albedo_prev1

            if self.recurrent:
                color_prev1 = output
            else:
                color_prev1 = color[i]

            albedo_prev1 = albedo[i]
            
            all_outputs.append(output)

        return torch.stack(all_outputs, dim=1)
