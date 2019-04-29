import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from modified_model import TemporalDenoiserModel
from modified_vanilla_model import VanillaDenoiserModelWrapper
from utils import pad_data, rgb, ycocg
from loss import Loss
from cyclic import CyclicLR
import os
import random
from itertools import chain, product
from filters import simple_filter, bilateral_filter
from smallmodel import SmallModel, KernelModel
import sys
import itertools

def prefilter_color(color, albedo):
    color = color / (albedo + 0.001)
    return simple_filter(color, factor=64)
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

def augment(color, normal, albedo, ref, prev1, prev2):
    if random.random() < 0.5:
        color = color.flip(-1)
        normal = normal.flip(-1)
        albedo = albedo.flip(-1)
        ref = ref.flip(-1)
        prev1 = prev1.flip(-1)
        prev2 = prev2.flip(-1)

    hsv_input = rgb_to_hsv(color)
    hsv_ref = rgb_to_hsv(ref)
    hsv_albedo = rgb_to_hsv(albedo)

    hue_shift = random.random() * 360

    hsv_input = torch.cat([(hsv_input[:, 0:1, ...] + hue_shift) % 360, hsv_input[:, 1:3, ...]], dim=1)
    hsv_ref = torch.cat([(hsv_ref[:, 0:1, ...] + hue_shift) % 360, hsv_ref[:, 1:3, ...]], dim=1)
    hsv_albedo = torch.cat([(hsv_albedo[:, 0:1, ...] + hue_shift) % 360, hsv_albedo[:, 1:3, ...]], dim=1)
    hsv_prev1 = torch.cat([(hsv_prev1[:, 0:1, ...] + hue_shift) % 360, hsv_prev1[:, 1:3, ...]], dim=1)
    hsv_prev2 = torch.cat([(hsv_prev2[:, 0:1, ...] + hue_shift) % 360, hsv_prev2[:, 1:3, ...]], dim=1)

    color = hsv_to_rgb(hsv_input)
    ref = hsv_to_rgb(hsv_ref)
    albedo = hsv_to_rgb(hsv_albedo)
    prev1 = hsv_to_rgb(hsv_prev1)
    prev2 = hsv_to_rgb(hsv_prev2)

    return color, normal, albedo, ref, prev1, prev2

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
        self.prev_irradiance1 = None
        self.prev_irradiance2 = None
        self.prev_ref_irradiance1 = None
        self.prev_ref_irradiance2 = None
        self.args = args

def train(state, color, normal, albedo, ref_color, alt_color, alt_ref):
    print(state.frame_num)
    total_loss = 0

    permutations = list(itertools.permutations([color, ref_color, alt_color, alt_ref], 2))

    for i in range(state.args.outer_train_iters):
        cur_color, cur_ref = permutations[i % 12]

        cur_ref_irradiance = cur_ref / (albedo + 0.001)
        stack = torch.cat([cur_color, normal, albedo, cur_ref_irradiance, state.prev_irradiance1, state.prev_irradiance2, state.prev_ref_irradiance1, state.prev_ref_irradiance2], dim=1)

        idxs = list(product(list(chain(range(0, color.shape[-1], state.args.cropsize)[:-1], [color.shape[-1] - state.args.cropsize])), list(chain(range(0, color.shape[-2], state.args.cropsize)[:-1], [color.shape[-2] - state.args.cropsize]))))

        if state.args.importance_sample:
            scored_idxs = [] 
            for x, y in idxs:
                irradiance_crop = state.prev_irradiance1[..., y:y+state.args.cropsize, x:x+state.args.cropsize]

                x_delta = (irradiance_crop[..., 0:state.args.cropsize - 1] - irradiance_crop[..., 1:state.args.cropsize]).abs().mean()
                y_delta = (irradiance_crop[..., 0:state.args.cropsize - 1, :] - irradiance_crop[..., 1:state.args.cropsize, :]).abs().mean()

                score = x_delta + y_delta
                scored_idxs.append((score, x, y))

            selected_idxs = [(x, y) for s, x, y in sorted(scored_idxs, reverse=True, key=lambda x: x[0])[0:state.args.num_crops*2]]
            selected_idxs = random.sample(selected_idxs, state.args.num_crops)
        else:
            selected_idxs = random.sample(idxs, state.args.num_crops)

        train_crops = []

        for x, y in selected_idxs:
            train_crops.append(stack[..., y:y+state.args.cropsize, x:x+state.args.cropsize])


        train_crops = torch.cat(train_crops, dim=0)

        if len(state.prev_crops) > 0:
            prev_crops = state.prev_crops
            train_crops = torch.cat([train_crops, prev_crops], dim=0)

            save_indices = np.random.permutation(state.args.num_crops * 2)[0:state.args.num_crops]
        else:
            save_indices = np.random.permutation(state.args.num_crops)

        state.prev_crops = train_crops[save_indices]


        color_train = train_crops[:, 0:3, ...]
        normal_train = train_crops[:, 3:5, ... ]
        albedo_train = train_crops[:, 5:8, ...]
        ref_irradiance_train = train_crops[:, 8:11, ...]
        prev_irradiance1_train = train_crops[:, 11:14, ...]
        prev_irradiance2_train = train_crops[:, 14:17, ...]
        prev_ref_irradiance1_train = train_crops[:, 17:20, ...]
        prev_ref_irradiance2_train = train_crops[:, 20:23, ...]

        if state.args.augment:
            color_train, normal_train, albedo_train, ref_irradiance_train, prev_irradiance1_train, prev_irradiance2_train = augment(color_train, normal_train, albedo_train, ref_irradiance_train, prev_irradiance1_train, prev_irradiance2_train, prev_ref_irradiance1_train, prev_ref_irradiance2_train)

        for i in range(state.args.inner_train_iters):
            output, e_irradiance = state.model(color_train, normal_train, albedo_train, prev_irradiance1_train, prev_irradiance2_train)
            state.optimizer.zero_grad()

            ref_irradiance_train = torch.stack([prev_ref_irradiance2_train, prev_ref_irradiance1_train, ref_irradiance_train], dim=1)
            e_irradiance = torch.stack([prev_irradiance2_train, prev_irradiance1_train, e_irradiance], dim=1)

            loss, _ = state.loss_gen.compute(ref_irradiance_train, e_irradiance)
            total_loss += loss
            loss.backward()
            state.optimizer.step()
            state.scheduler.batch_step()

    #state.scheduler.step()
    state.frame_num += 1
    print(float(loss.cpu()) / state.args.outer_train_iters)

def create_model(args, dev, weights):
    model = KernelModel().to(dev)
    #model = VanillaDenoiserModelWrapper(init=True).to(dev)

    if weights is not None:
        state_dict = torch.load(weights, map_location='cpu')

        state_dict = { '.'.join(k.split('.')[1:]): v for k, v in state_dict.items() }

        model.load_state_dict(state_dict)

    return model

def init_training_state(dev=torch.device('cuda:{}'.format(0)), init_weights=None):
    #args = Args(lr=0.001, outer_train_iters=1, inner_train_iters=1, num_crops=32, cropsize=64, augment=False, importance_sample=False)
    args = Args(lr=0.0003, outer_train_iters=128, inner_train_iters=1, num_crops=8, cropsize=128, augment=False, importance_sample=False)
    model = create_model(args, dev, init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
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

def train_and_eval(training_state, color, ref_color, normal, albedo, alt_color, alt_ref, crop=True):
    color[torch.isnan(color)] = 0
    ref_color[torch.isnan(ref_color)] = 0

    ref_e_irradiance = ref_color / (albedo + 0.001)

    if crop:
        orig_color = color 
        orig_normal = normal
        orig_albedo = albedo
    
        color = color[..., 480:960+480]
        ref_color = ref_color[..., 480:960+480]
        normal = normal[..., 480:960+480]
        albedo = albedo[..., 480:960+480]

    color_pad, normal_pad, albedo_pad = pad_data(color), pad_data(normal), pad_data(albedo)

    if training_state.prev_irradiance1 is None:
        training_state.prev_irradiance1 = torch.zeros_like(color_pad)
        training_state.prev_irradiance2 = training_state.prev_irradiance1
        training_state.prev_ref_irradiance1 = training_state.prev_irradiance1
        training_state.prev_ref_irradiance2 = training_state.prev_irradiance1

    height, width = color.shape[-2:]
    train(training_state, color, normal, albedo, ref_color, alt_color, alt_ref)

    with torch.no_grad():
        output, e_irradiance = training_state.model(color_pad, normal_pad, albedo_pad, training_state.prev_irradiance1, training_state.prev_irradiance2)
        training_state.prev_irradiance2 = training_state.prev_irradiance1
        training_state.prev_irradiance1 = e_irradiance

        training_state.prev_ref_irradiance2 = training_state.prev_ref_irradiance1
        training_state.prev_ref_irradiance1 = ref_e_irradiance

        output = output[..., 0:height, 0:width]

        if crop:
            right = orig_color[..., 960+480:1920]
            right_normal = orig_normal[..., 960+480:1920]
            right_albedo = orig_albedo[..., 960+480:1920]
            filtered_right = prefilter_color(pad_data(right, mul=64), pad_data(right_albedo, mul=64))[:, :, 0:right.shape[2], 0:right.shape[3]] * (right_albedo + 0.001)

            output = torch.cat([orig_color[..., 0:480], output, filtered_right], dim=-1)

    return output
