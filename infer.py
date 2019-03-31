import torch
import torchvision
from dataset import ExrDataset
from state import StateManager
from arg_handler import parse_infer_args
from modified_model import TemporalDenoiserModel
from modified_vanilla_model import TemporalVanillaDenoiserModel
from utils import tonemap
from data_loading import pad_data, save_exr, save_png
from filters import simple_filter
import os

args = parse_infer_args()
dev = torch.device('cuda:{}'.format(args.gpu))
if args.vanilla_net:
    model = TemporalVanillaDenoiserModel(init=False).to(dev)
else:
    model = TemporalDenoiserModel(recurrent=not args.disable_recurrence, init=False).to(dev)

model.load_state_dict(torch.load(args.weights, map_location='cpu'))
model.eval()

dataset = ExrDataset(dataset_path=args.inputs,
                     training=False,
                     num_imgs=args.num_imgs)

for i in range(args.start_frame, len(dataset)):
    color, normal, albedo, direct, indirect, tshadow = dataset[i]
    color, normal, albedo, direct, indirect, tshadow = color.to(dev), normal.to(dev), albedo.to(dev), direct.to(dev), indirect.to(dev), tshadow.to(dev)
    color, normal, albedo, direct, indirect, tshadow = pad_data(color), pad_data(normal), pad_data(albedo), pad_data(direct), pad_data(indirect), pad_data(tshadow)
    color, normal, albedo, direct, indirect, tshadow = color.unsqueeze(dim=0), normal.unsqueeze(dim=0), albedo.unsqueeze(dim=0), direct.unsqueeze(dim=0), indirect.unsqueeze(dim=0), tshadow.unsqueeze(dim=0)

    with torch.no_grad():
        prefiltered = simple_filter(color, factor=64)
        output, e_irradiance = model(color.unsqueeze(dim=1), normal.unsqueeze(dim=1), albedo.unsqueeze(dim=1),
                                     prefiltered.unsqueeze(dim=1))

        output = output[..., 0:args.img_height, 0:args.img_width]
        e_irradiance = e_irradiance[..., 0:args.img_height, 0:args.img_width]
        output = output.squeeze()
        e_irradiance = e_irradiance.squeeze()

        if args.disable_recurrence:
            color_prev1 = color
        else:
            color_prev1 = output

        color_prev2 = color_prev1

    save_exr(output, os.path.join(args.outputs, 'out_{}.exr'.format(i)))
    save_exr(e_irradiance, os.path.join(args.outputs, 'ei_{}.exr'.format(i)))

    save_png(output, os.path.join(args.outputs, 'out_{}.png'.format(i)))
