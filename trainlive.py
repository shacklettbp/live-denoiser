import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from dataset import ExrDataset
from state import StateManager
from arg_handler import parse_infer_args
from modified_model import TemporalDenoiserModel
from modified_vanilla_model import TemporalVanillaDenoiserModel
from utils import tonemap
from data_loading import pad_data, save_exr, save_png
from loss import Loss
import os
import random
from itertools import chain, product

args = parse_infer_args()
dev = torch.device('cuda:{}'.format(args.gpu))
if args.vanilla_net:
    model = TemporalVanillaDenoiserModel(init=True).to(dev)
else:
    #model = TemporalDenoiserModel(recurrent=not args.disable_recurrence, init=True).to(dev)
    class Conv(nn.Module):
        def __init__(self, in_channels, out_channels, relu=True):
            super(Conv, self).__init__()
            arr = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)]
            if relu:
                arr.append(nn.ReLU())

            self.model = nn.Sequential(*arr)

        def forward(self, input):
            return self.model(input)

    class UNet(nn.Module):
        def __init__(self):
            super(UNet, self).__init__()
            self.enc1 = Conv(9, 32)
            self.enc2 = Conv(32, 64)
            self.enc3 = Conv(64, 96)
            self.enc4 = Conv(96, 128)

            self.dec4 = Conv(128, 96)
            self.dec3 = Conv(96*2, 64)
            self.dec2 = Conv(64*2, 32)
            self.dec1 = Conv(32*2, 16)
            self.final = Conv(16, 3, False)

        def forward(self, input):
            enc1_out = self.enc1(input)
            enc2_out = self.enc2(F.interpolate(enc1_out, scale_factor=0.5))
            enc3_out = self.enc3(F.interpolate(enc2_out, scale_factor=0.5))
            enc4_out = self.enc4(F.interpolate(enc3_out, scale_factor=0.5))

            out = self.dec4(enc4_out)
            out = F.interpolate(out, scale_factor=2)
            out = self.dec3(torch.cat([out, enc3_out], dim=1))
            out = F.interpolate(out, scale_factor=2)
            out = self.dec2(torch.cat([out, enc2_out], dim=1))
            out = F.interpolate(out, scale_factor=2)
            out = self.dec1(torch.cat([out, enc1_out], dim=1))

            out = self.final(out)

            return out

    class SmallDenoiserModel(nn.Module):
        def __init__(self):
            super(SmallDenoiserModel, self).__init__()
            self.model = UNet()

        def forward(self, color, normal, albedo, direct, indirect, tshadow):
            eps = 0.001
            color = color.squeeze(dim=1)
            normal = normal.squeeze(dim=1)
            albedo = albedo.squeeze(dim=1)

            color = color / (albedo + eps)

            full_input = torch.cat([color, normal, albedo], dim=1)

            output = self.model(full_input)

            exp = torch.expm1(output)

            return exp * (albedo + eps), exp

    model = SmallDenoiserModel().to(dev)
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
    model.apply(init_weights)

#model.load_state_dict(torch.load("ref_weights/test.pth", map_location='cpu'))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
loss_gen = Loss(dev)

dataset = ExrDataset(dataset_path=args.inputs,
                     training=True,
                     num_imgs=args.num_imgs)

full_image = False

outer_train_iters = 1
inner_train_iters = 8
num_crops = 16

def train(color, normal, albedo, ref, direct, indirect, tshadow):
    for i in range(outer_train_iters):
        cropsize = 128
        crops = []

        idxs = list(product(list(chain(range(0, 1920, cropsize)[:-1], [1920 - cropsize])), list(chain(range(0, 1080, cropsize)[:-1], [1080 - cropsize]))))

        rand_idxs = random.sample(idxs, num_crops)

        color_train = []
        normal_train = []
        albedo_train = []
        ref_train = []
        direct_train = []
        indirect_train = []
        tshadow_train = []

        for x, y in rand_idxs:
            color_crop = color[..., y:y+cropsize, x:x+cropsize]
            normal_crop = normal[..., y:y+cropsize, x:x+cropsize]
            albedo_crop = albedo[..., y:y+cropsize, x:x+cropsize]
            ref_crop = ref[..., y:y+cropsize, x:x+cropsize]
            direct_crop = direct[..., y:y+cropsize, x:x+cropsize]
            indirect_crop = indirect[..., y:y+cropsize, x:x+cropsize]
            tshadow_crop = tshadow[..., y:y+cropsize, x:x+cropsize]

            color_indices = np.random.permutation(3)

            color_crop = color_crop[:, color_indices, ...]
            albedo_crop = albedo_crop[:, color_indices, ...]
            ref_crop = ref_crop[:, color_indices, ...]
            direct_crop = direct_crop[:, color_indices, ...]
            indirect_crop = indirect_crop[:, color_indices, ...]

            color_train.append(color_crop)
            normal_train.append(normal_crop)
            albedo_train.append(albedo_crop)
            ref_train.append(ref_crop)
            direct_train.append(direct_crop)
            indirect_train.append(indirect_crop)
            tshadow_train.append(tshadow_crop)

        color_train = torch.cat(color_train)
        normal_train = torch.cat(normal_train)
        albedo_train = torch.cat(albedo_train)
        ref_train = torch.cat(ref_train)
        direct_train = torch.cat(direct_train)
        indirect_train = torch.cat(indirect_train)
        tshadow_train = torch.cat(tshadow_train)

        for i in range(inner_train_iters):
            output, e_irradiance = model(color_train.unsqueeze(dim=1), normal_train.unsqueeze(dim=1), albedo_train.unsqueeze(dim=1), direct_train.unsqueeze(dim=1), indirect_train.unsqueeze(dim=1), tshadow_train.unsqueeze(dim=1))

            optimizer.zero_grad()
            loss, _ = loss_gen.compute(output, ref_train, color_train, albedo_train, e_irradiance)
            loss.backward()
            optimizer.step()

out_num = 0
while True:
    for i in range(0, 1800):
        color, normal, albedo, ref, direct, indirect, tshadow = dataset[i]
        color, normal, albedo, ref, direct, indirect, tshadow = color.to(dev), normal.to(dev), albedo.to(dev), ref.to(dev), direct.to(dev), indirect.to(dev), tshadow.to(dev)
        color, normal, albedo, ref, direct, indirect, tshadow = pad_data(color), pad_data(normal), pad_data(albedo), pad_data(ref), pad_data(direct), pad_data(indirect), pad_data(tshadow)
        color, normal, albedo, ref, direct, indirect, tshadow = color.unsqueeze(dim=0), normal.unsqueeze(dim=0), albedo.unsqueeze(dim=0), ref.unsqueeze(dim=0), direct.unsqueeze(dim=0), indirect.unsqueeze(dim=0), tshadow.unsqueeze(dim=0)
    
        train(color, normal, albedo, ref, direct, indirect, tshadow)
    
        output, e_irradiance = model(color.unsqueeze(dim=1), normal.unsqueeze(dim=1), albedo.unsqueeze(dim=1),
                                     direct.unsqueeze(dim=1), indirect.unsqueeze(dim=1), tshadow.unsqueeze(dim=1))
    
        output = output.detach()[..., 0:args.img_height, 0:args.img_width]
        e_irradiance = e_irradiance.detach()[..., 0:args.img_height, 0:args.img_width]
        output = output.squeeze()
        e_irradiance = e_irradiance.squeeze()
    
        save_exr(output, os.path.join(args.outputs, 'out_{}.exr'.format(out_num)))
        out_num += 1
        #save_exr(e_irradiance, os.path.join(args.outputs, 'ei_{}.exr'.format(i)))
    
        #save_png(output, os.path.join(args.outputs, 'out_{}.png'.format(i)))
