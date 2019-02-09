import torch
import torchvision
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
from itertools import chain

args = parse_infer_args()
dev = torch.device('cuda:{}'.format(args.gpu))
if args.vanilla_net:
    model = TemporalVanillaDenoiserModel(init=True).to(dev)
else:
    model = TemporalDenoiserModel(recurrent=not args.disable_recurrence, init=True).to(dev)

model.load_state_dict(torch.load("ref_weights/test.pth", map_location='cpu'))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
loss_gen = Loss(dev)

dataset = ExrDataset(dataset_path=args.inputs,
                     training=True,
                     num_imgs=args.num_imgs)

full_image = True

def train(output, ref, input_color, input_albedo, output_irradiance):
    if full_image:
        optimizer.zero_grad()
        loss, _ = loss_gen.compute(output, ref, input_color, input_albedo, output_irradiance)
        loss.backward()
        optimizer.step()
    else:
        cropsize = 256
        crops = []
        for y in range(0, 1080, cropsize)[:-1]:
            for x in range(0, 1920, cropsize)[:-1]:
                out_crop = output[..., y:y+cropsize, x:x+cropsize]
                ref_crop = ref[..., y:y+cropsize, x:x+cropsize]
                color_crop = input_color[..., y:y+cropsize, x:x+cropsize]
                albedo_crop = input_albedo[..., y:y+cropsize, x:x+cropsize]
                irradiance_crop = output_irradiance[..., y:y+cropsize, x:x+cropsize]

                d_x = (irradiance_crop[..., 0:cropsize-1] - irradiance_crop[..., 1:cropsize])**2
                d_y = (irradiance_crop[..., 0:cropsize-1, :] - irradiance_crop[..., 1:cropsize, :])**2
                score = d_x.mean() + d_y.mean()

                crops.append((score, out_crop, ref_crop, color_crop, albedo_crop, irradiance_crop))

        #train_crops = random.sample(crops, 16)
        train_crops = sorted(crops, key=lambda x: x[0])[::-1][:16]
        out_train = []
        ref_train = []
        color_train = []
        albedo_train = []
        irradiance_train = []

        for _, out_crop, ref_crop, color_crop, albedo_crop, irradiance_crop in train_crops:
            out_train.append(out_crop)
            ref_train.append(ref_crop)
            color_train.append(color_crop)
            albedo_train.append(albedo_crop)
            irradiance_train.append(irradiance_crop)

        optimizer.zero_grad()
        loss, _ = loss_gen.compute(torch.cat(out_train), torch.cat(ref_train), torch.cat(color_train), torch.cat(albedo_train), torch.cat(irradiance_train))
        loss.backward()
        optimizer.step()

for i in range(args.start_frame, len(dataset)):
    color, normal, albedo, ref, direct, indirect, tshadow = dataset[i]
    color, normal, albedo, ref, direct, indirect, tshadow = color.to(dev), normal.to(dev), albedo.to(dev), ref.to(dev), direct.to(dev), indirect.to(dev), tshadow.to(dev)
    color, normal, albedo, ref, direct, indirect, tshadow = pad_data(color), pad_data(normal), pad_data(albedo), pad_data(ref), pad_data(direct), pad_data(indirect), pad_data(tshadow)
    color, normal, albedo, ref, direct, indirect, tshadow = color.unsqueeze(dim=0), normal.unsqueeze(dim=0), albedo.unsqueeze(dim=0), ref.unsqueeze(dim=0), direct.unsqueeze(dim=0), indirect.unsqueeze(dim=0), tshadow.unsqueeze(dim=0)

    output, e_irradiance = model(color.unsqueeze(dim=1), normal.unsqueeze(dim=1), albedo.unsqueeze(dim=1),
                                 direct.unsqueeze(dim=1), indirect.unsqueeze(dim=1), tshadow.unsqueeze(dim=1))

    train(output, ref, color, albedo, e_irradiance)

    output = output.detach()[..., 0:args.img_height, 0:args.img_width]
    e_irradiance = e_irradiance.detach()[..., 0:args.img_height, 0:args.img_width]
    output = output.squeeze()
    e_irradiance = e_irradiance.squeeze()

    save_exr(output, os.path.join(args.outputs, 'out_{}.exr'.format(i)))
    #save_exr(e_irradiance, os.path.join(args.outputs, 'ei_{}.exr'.format(i)))

    save_png(output, os.path.join(args.outputs, 'out_{}.png'.format(i)))
