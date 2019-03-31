import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from modified_model import TemporalDenoiserModel
from modified_vanilla_model import TemporalVanillaDenoiserModel
from utils import pad_data, rgb, ycocg
from loss import Loss
from cyclic import CyclicLR
import os
import random
from itertools import chain, product
from filters import simple_filter, bilateral_filter
import sys

def prefilter_color(*args, **kwargs):
    return simple_filter(*args, **kwargs, factor=64)
    #return bilateral_filter(*args, **kwargs)

def rgb_to_hsv(color):
    r = color[:, 0:1, ...]
    g = color[:, 1:2, ...]
    b = color[:, 2:3, ...]

    c_max = torch.max(color, dim=1, keepdim=True)[0]
    c_min = torch.min(color, dim=1, keepdim=True)[0]

    d = c_max - c_min
    v = c_max
    s = torch.where(c_max > 0, d / c_max, torch.zeros_like(d))

    h = torch.where(r == c_max, g - b, torch.where(g == c_max, 2 + b - r, 4 + r - g))
    h = torch.where(d > 0, h / d * 60, torch.zeros_like(h)) % 360

    return torch.cat([h, s, v], dim=1)

def hsv_to_rgb(color):
    h = color[:, 0:1, ...]
    s = color[:, 1:2, ...]
    v = color[:, 2:3, ...]

    h = h / 60
    c = v * s
    x = c * (1 - (h % 2 - 1).abs())
    m = v - c
    z = torch.zeros_like(c)

    rgb = torch.where(h <= 1, torch.cat([c, x, z], dim=1),
            torch.where(h <= 2, torch.cat([x, c, z], dim=1),
              torch.where(h <= 3, torch.cat([z, c, x], dim=1),
                torch.where(h <= 4, torch.cat([z, x, c], dim=1),
                  torch.where(h <= 5, torch.cat([x, z, c], dim=1),
                    torch.cat([c, z, x], dim=1))))))

    rgb = rgb + m

    return rgb

def augment(color, normal, albedo, ref):
    if random.random() < 0.5:
        color = color.flip(-1)
        normal = normal.flip(-1)
        albedo = albedo.flip(-1)
        ref = ref.flip(-1)

    hsv_input = rgb_to_hsv(color)
    hsv_ref = rgb_to_hsv(ref)
    hsv_albedo = rgb_to_hsv(albedo)

    hue_shift = random.random() * 360

    hsv_input = torch.cat([(hsv_input[:, 0:1, ...] + hue_shift) % 360, hsv_input[:, 1:3, ...]], dim=1)
    hsv_ref = torch.cat([(hsv_ref[:, 0:1, ...] + hue_shift) % 360, hsv_ref[:, 1:3, ...]], dim=1)
    hsv_albedo = torch.cat([(hsv_albedo[:, 0:1, ...] + hue_shift) % 360, hsv_albedo[:, 1:3, ...]], dim=1)

    color = hsv_to_rgb(hsv_input)
    ref = hsv_to_rgb(hsv_ref)
    albedo = hsv_to_rgb(hsv_albedo)

    return color, normal, albedo, ref

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class TrainingState:
    def __init__(self, model, optimizer, scheduler, loss_gen, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_gen = loss_gen
        self.frame_num = 0
        self.prev_crops = ()
        self.prev_irradiance = None
        self.args = args

def train(state, color, normal, albedo, ref):
    print(state.frame_num)
    for i in range(state.args.outer_train_iters):
        crops = []

        idxs = list(product(list(chain(range(0, 960, state.args.cropsize)[:-1], [960 - state.args.cropsize])), list(chain(range(0, 1080, state.args.cropsize)[:-1], [1080 - state.args.cropsize]))))

        if state.args.importance_sample:
            scored_idxs = [] 
            for x, y in idxs:
                irradiance_crop = state.prev_irradiance[..., y:y+state.args.cropsize, x:x+state.args.cropsize]

                x_delta = (irradiance_crop[..., 0:state.args.cropsize - 1] - irradiance_crop[..., 1:state.args.cropsize]).abs().mean()
                y_delta = (irradiance_crop[..., 0:state.args.cropsize - 1, :] - irradiance_crop[..., 1:state.args.cropsize, :]).abs().mean()

                score = x_delta + y_delta
                scored_idxs.append((score, x, y))

            selected_idxs = [(x, y) for s, x, y in sorted(scored_idxs, reverse=True, key=lambda x: x[0])[0:state.args.num_crops*2]]
            selected_idxs = random.sample(selected_idxs, state.args.num_crops)
        else:
            selected_idxs = random.sample(idxs, state.args.num_crops)

        color_train = []
        normal_train = []
        albedo_train = []
        ref_train = []

        for x, y in selected_idxs:
            color_crop = color[..., y:y+state.args.cropsize, x:x+state.args.cropsize]
            normal_crop = normal[..., y:y+state.args.cropsize, x:x+state.args.cropsize]
            albedo_crop = albedo[..., y:y+state.args.cropsize, x:x+state.args.cropsize]
            ref_crop = ref[..., y:y+state.args.cropsize, x:x+state.args.cropsize]

            color_train.append(color_crop)
            normal_train.append(normal_crop)
            albedo_train.append(albedo_crop)
            ref_train.append(ref_crop)

        color_train = torch.cat(color_train)
        normal_train = torch.cat(normal_train)
        albedo_train = torch.cat(albedo_train)
        ref_train = torch.cat(ref_train)

        if len(state.prev_crops) == 4:
            color_train_prev, normal_train_prev, albedo_train_prev, ref_train_prev = state.prev_crops

            color_train = torch.cat([color_train, color_train_prev])
            normal_train = torch.cat([normal_train, normal_train_prev])
            albedo_train = torch.cat([albedo_train, albedo_train_prev])
            ref_train = torch.cat([ref_train, ref_train_prev])
            save_indices = np.random.permutation(state.args.num_crops * 2)
        else:
            save_indices = np.random.permutation(state.args.num_crops)

            
        state.prev_crops = (color_train[save_indices], normal_train[save_indices], albedo_train[save_indices], ref_train[save_indices])


        if state.args.augment:
            color_train, normal_train, albedo_train, ref_train = augment(color_train, normal_train, albedo_train, ref_train)


        prefiltered_train = prefilter_color(color_train)

        for i in range(state.args.inner_train_iters):
            output, e_irradiance = state.model(color_train.unsqueeze(dim=1), normal_train.unsqueeze(dim=1), albedo_train.unsqueeze(dim=1), prefiltered_train.unsqueeze(dim=1))
            output = output.squeeze(dim=1)
            e_irradiance = e_irradiance.squeeze(dim=1)

            state.optimizer.zero_grad()
            loss, _ = state.loss_gen.compute(output, ref_train, color_train, albedo_train, e_irradiance)
            loss.backward()
            state.optimizer.step()
            state.scheduler.batch_step()

    #state.scheduler.step()
    state.frame_num += 1

def create_model(dev):
    return TemporalVanillaDenoiserModel(init=True).to(dev)

    class ModelImpl(nn.Module):
        def __init__(self):
            super(ModelImpl, self).__init__()

            self.start = nn.Sequential(
                    nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU())

            self.enc = nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU())


            self.dec = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU())

            self.final = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0)

        def forward(self, full_input):
            full_input = full_input[:, 0:9, ...]
            normal = full_input[:, 3:6, ...]

            start = self.start(full_input)

            enc1_out = self.enc(start)
            out = F.avg_pool2d(enc1_out, kernel_size=2, stride=2)

            enc2_out = self.enc(out)
            out = F.avg_pool2d(enc2_out, kernel_size=2, stride=2)

            enc3_out = self.enc(out)
            out = F.avg_pool2d(enc3_out, kernel_size=2, stride=2)

            enc4_out = self.enc(out)
            out = enc4_out

            out = F.interpolate(out, scale_factor=2, mode='bilinear')
            #out = self.dec(torch.cat([out, F.interpolate(start, scale_factor=1/4, mode='bilinear')], dim=1))
            out = self.dec(torch.cat([out, enc3_out], dim=1))

            out = F.interpolate(out, scale_factor=2, mode='bilinear')
            #out = self.dec(torch.cat([out, F.interpolate(start, scale_factor=1/2, mode='bilinear')], dim=1))
            out = self.dec(torch.cat([out, enc2_out], dim=1))

            out = F.interpolate(out, scale_factor=2, mode='bilinear')
            #out = self.dec(torch.cat([out, start], dim=1))
            out = self.dec(torch.cat([out, enc1_out], dim=1))

            return self.final(out)

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

            self.model = ModelImpl()

        def forward(self, color, normal, albedo, prefiltered):
            color = color.squeeze(dim=1)
            normal = normal.squeeze(dim=1)
            albedo = albedo.squeeze(dim=1)
            prefiltered = prefiltered.squeeze(dim=1)

            eps = 0.001
            color = color / (albedo + eps)

            mapped_color = torch.log1p(color)
            mapped_albedo = torch.log1p(albedo)
            mapped_prefiltered = torch.log1p(prefiltered)

            full_input = torch.cat([mapped_color, normal, mapped_albedo, mapped_prefiltered], dim=1)

            out = self.model(full_input)

            exp = torch.expm1(out)

            return exp * (albedo + eps), exp

    model = Model()
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)

    model.apply(init_weights)

    model = model.to(dev)

    return model

def init_training_state():
    args = Args(lr=0.001, outer_train_iters=1, inner_train_iters=1, num_crops=16, cropsize=128, augment=False, importance_sample=False)
    dev = torch.device('cuda:{}'.format(0))
    model = create_model(dev)
    #model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "weights_1000.pth"), map_location='cpu'))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    loss_gen = Loss(dev)

    def schedule_func(frame_num):
        print(frame_num)
        return args.lr
        if frame_num < 10:
            return 0.0003
        if frame_num < 1000:
            return 0.0003
        elif frame_num < 2000:
            return 0.0001
        elif frame_num < 3000:
            return 0.001**(frame_num/200)
        else:
            return 5e-5

    #scheduler = LambdaLR(optimizer, lr_lambda=schedule_func)
    scheduler = CyclicLR(optimizer, args.lr / 10, args.lr, step_size=50)

    return TrainingState(model, optimizer, scheduler, loss_gen, args)

def train_and_eval(training_state, color, ref_color, normal, albedo):
    orig_color = color 
    orig_normal = normal

    color = color[..., 480:960+480]
    ref_color = ref_color[..., 480:960+480]
    normal = normal[..., 480:960+480]
    albedo = albedo[..., 480:960+480]

    if training_state.prev_irradiance is None:
        training_state.prev_irradiance = color

    train(training_state, color, normal, albedo, ref_color)
    
    height, width = color.shape[-2:]

    color_pad, normal_pad, albedo_pad = pad_data(color), pad_data(normal), pad_data(albedo)

    with torch.no_grad():
        prefiltered_color = prefilter_color(color_pad)
        output, e_irradiance = training_state.model(color_pad.unsqueeze(dim=1), normal_pad.unsqueeze(dim=1), albedo_pad.unsqueeze(dim=1), prefiltered_color.unsqueeze(dim=1))
        output = output.squeeze(dim=1)
        output = output[..., 0:height, 0:width]

        right = orig_color[..., 960+480:1920]
        right_normal = orig_normal[..., 960+480:1920]
        filtered_right = prefilter_color(pad_data(right, mul=64))[:, :, 0:right.shape[2], 0:right.shape[3]]

        output = torch.cat([orig_color[..., 0:480], output, filtered_right], dim=-1)

    training_state.prev_irradiance = e_irradiance

    return output
