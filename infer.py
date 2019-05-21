import torch
import torchvision
from dataset import ExrDataset
from state import StateManager
from arg_handler import parse_infer_args
from modified_model import DenoiserModel
from modified_vanilla_model import VanillaDenoiserModel
from smallmodel import KernelModel
from hierarchicalmodel import HierarchicalKernelModel
from utils import tonemap
from data_loading import pad_data, save_exr, save_png
from filters import simple_filter
import os

args = parse_infer_args()
dev = torch.device('cuda:{}'.format(args.gpu))
if args.vanilla_net:
    model = KernelModel().to(dev)
else:
    model = HierarchicalKernelModel().to(dev)

state_dict = torch.load(args.weights, map_location='cpu')
state_dict = { '.'.join(k.split('.')[1:]): v for k, v in state_dict.items() }
model.load_state_dict(state_dict)
model.eval()

dataset = ExrDataset(dataset_path=args.inputs,
                     num_imgs=args.num_imgs)

color_prev1 = torch.zeros_like(pad_data(dataset[0][0].to(dev)).unsqueeze(dim=0))
color_prev2 = torch.zeros_like(color_prev1)

if not os.path.isdir(args.outputs):
    os.makedirs(args.outputs, exist_ok = True)

for i in range(args.start_frame, len(dataset)):
    color, normal, albedo = dataset[i]
    color, normal, albedo = color.to(dev), normal.to(dev), albedo.to(dev)
    color, normal, albedo = pad_data(color), pad_data(normal), pad_data(albedo)
    color, normal, albedo = color.unsqueeze(dim=0), normal.unsqueeze(dim=0), albedo.unsqueeze(dim=0)

    with torch.no_grad():
        output, e_irradiance, albedo_output = model(color, normal, albedo, color_prev1, color_prev2)

        color_prev2 = color_prev1
        if args.disable_recurrence:
            color_prev1 = color
        else:
            color_prev1 = e_irradiance

        output = output[..., 0:args.img_height, 0:args.img_width]
        e_irradiance = e_irradiance[..., 0:args.img_height, 0:args.img_width]
        output = output.squeeze(dim=0)
        e_irradiance = e_irradiance.squeeze(dim=0)

        albedo_output = albedo_output[..., 0:args.img_height, 0:args.img_width]
        albedo_output = albedo_output.squeeze(dim=0)

    save_exr(output, os.path.join(args.outputs, 'out_{}.exr'.format(i)))
    save_exr(albedo_output, os.path.join(args.outputs, 'albedo_out_{}.exr'.format(i)))
    save_exr(e_irradiance, os.path.join(args.outputs, 'ei_{}.exr'.format(i)))

    #save_png(output, os.path.join(args.outputs, 'out_{}.png'.format(i)))
