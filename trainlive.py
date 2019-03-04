import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from modified_model import TemporalDenoiserModel
from modified_vanilla_model import TemporalVanillaDenoiserModel
from utils import pad_data
from loss import Loss
from cyclic import CyclicLR
import os
import random
from itertools import chain, product

def augment(color, normal, albedo, ref):
    return color, normal, albedo, ref
    color_indices = np.random.permutation(3)

    color = color[:, color_indices, ...]
    albedo = albedo[:, color_indices, ...]
    ref = ref[:, color_indices, ...]

    return color, normal, albedo, ref

def ycocg(tensor):
    return tensor
    r = tensor[:, 0, ...]
    g = tensor[:, 1, ...]
    b = tensor[:, 2, ...]

    y = r / 4 + g / 2 + b / 4
    co = r / 2 - b / 2
    cg = -r / 4 + g / 2 - b / 4

    assert((y >= 0).all())
    assert((co >= 0).all())
    assert((cg >= 0).all())

    return torch.stack([y, co, cg], dim=1)

def rgb(tensor):
    return tensor

    y = tensor[:, 0, ...]
    co = tensor[:, 1, ...]
    cg = tensor[:, 2, ...]

    r = y + co - cg
    g = y + cg
    b = y - co - cg

    return torch.stack([r, g, b], dim=1)

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


        color_train = ycocg(color_train)
        albedo_train = ycocg(albedo_train)
        ref_train = ycocg(ref_train)

        for i in range(state.args.inner_train_iters):
            output, e_irradiance = state.model(color_train.unsqueeze(dim=1), normal_train.unsqueeze(dim=1), albedo_train.unsqueeze(dim=1))
            output = output.squeeze(dim=1)
            e_irradiance = e_irradiance.squeeze(dim=1)

            state.optimizer.zero_grad()
            loss, _ = state.loss_gen.compute(output, ref_train, color_train, albedo_train, e_irradiance)
            loss.backward()
            state.optimizer.step()
            state.scheduler.batch_step()

    #state.scheduler.step()
    state.frame_num += 1

def init_training_state():
    args = Args(lr=0.001, outer_train_iters=1, inner_train_iters=1, num_crops=16, cropsize=256)
    dev = torch.device('cuda:{}'.format(0))
    model = TemporalVanillaDenoiserModel(init=True).to(dev)
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

    color = color[..., 480:960+480]
    ref_color = ref_color[..., 480:960+480]
    normal = normal[..., 480:960+480]
    albedo = albedo[..., 480:960+480]

    train(training_state, color, normal, albedo, ref_color)
    
    height, width = color.shape[-2:]

    color_pad, normal_pad, albedo_pad = pad_data(color), pad_data(normal), pad_data(albedo)

    with torch.no_grad():
        output, e_irradiance = training_state.model(ycocg(color_pad).unsqueeze(dim=1), normal_pad.unsqueeze(dim=1), ycocg(albedo_pad).unsqueeze(dim=1))
        output = rgb(output.squeeze(dim=1))
        output = output[..., 0:height, 0:width]

    output = torch.cat([orig_color[..., 0:480], output, orig_color[..., 960+480:1920]], dim=-1)

    return output
