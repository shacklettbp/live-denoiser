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

def augment(color, normal, albedo, ref):
    return color, normal, albedo, ref
    color_indices = np.random.permutation(3)

    color = color[:, color_indices, ...]
    albedo = albedo[:, color_indices, ...]
    ref = ref[:, color_indices, ...]

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
        self.args = args

def train(state, color, normal, albedo, ref):
    print(state.frame_num)
    for i in range(state.args.outer_train_iters):
        crops = []

        idxs = list(product(list(chain(range(0, 960, state.args.cropsize)[:-1], [960 - state.args.cropsize])), list(chain(range(0, 1080, state.args.cropsize)[:-1], [1080 - state.args.cropsize]))))

        rand_idxs = random.sample(idxs, state.args.num_crops)

        color_train = []
        normal_train = []
        albedo_train = []
        ref_train = []

        for x, y in rand_idxs:
            color_crop = color[..., y:y+state.args.cropsize, x:x+state.args.cropsize]
            normal_crop = normal[..., y:y+state.args.cropsize, x:x+state.args.cropsize]
            albedo_crop = albedo[..., y:y+state.args.cropsize, x:x+state.args.cropsize]
            ref_crop = ref[..., y:y+state.args.cropsize, x:x+state.args.cropsize]

            color_crop, normal_crop, albedo_crop, ref_crop = augment(color_crop, normal_crop, albedo_crop, ref_crop)

            color_train.append(color_crop)
            normal_train.append(normal_crop)
            albedo_train.append(albedo_crop)
            ref_train.append(ref_crop)

        color_train = torch.cat(color_train)
        normal_train = torch.cat(normal_train)
        albedo_train = torch.cat(albedo_train)
        ref_train = torch.cat(ref_train)

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
    args = Args(lr=0.001, outer_train_iters=1, inner_train_iters=1, num_crops=32, cropsize=128)
    dev = torch.device('cuda:{}'.format(0))
    model = create_model(dev)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "weights_1000.pth"), map_location='cpu'))
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
    assert((color != ref_color).any())
    orig_color = color 
    orig_normal = normal

    color = color[..., 480:960+480]
    ref_color = ref_color[..., 480:960+480]
    normal = normal[..., 480:960+480]
    albedo = albedo[..., 480:960+480]

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

    return output
