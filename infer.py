import torch
import torchvision
from dataset import ExrDataset
from state import StateManager
from arg_handler import parse_infer_args
from modified_model import TemporalDenoiserModel
from modified_vanilla_model import TemporalVanillaDenoiserModel
from utils import tonemap
from data_loading import pad_data, save_exr, save_png
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
    color, normal, albedo = dataset[i]
    color, normal, albedo = color.to(dev), normal.to(dev), albedo.to(dev)
    color, normal, albedo = pad_data(color), pad_data(normal), pad_data(albedo)
    color, normal, albedo = color.unsqueeze(dim=0), normal.unsqueeze(dim=0), albedo.unsqueeze(dim=0)

    color_prev1 = torch.zeros_like(color)
    color_prev2 = torch.zeros_like(color)

    with torch.no_grad():
        output, e_irradiance = model(color.unsqueeze(dim=1), normal.unsqueeze(dim=1), albedo.unsqueeze(dim=1),
                                      color_prev1=color_prev1, color_prev2=color_prev2)

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
