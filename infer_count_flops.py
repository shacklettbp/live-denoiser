import torch
import torchvision
from dataset import ExrDataset
from state import StateManager
from arg_handler import parse_infer_args
from model import DenoiserModel
from vanilla_model import VanillaDenoiserModel
from utils import tonemap
from data_loading import pad_data, save_exr
import os
from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

args = parse_infer_args()
dev = torch.device('cuda:{}'.format(args.gpu))
if args.vanilla_net:
    model = VanillaDenoiserModel(init=False).to(dev)
else:
    model = DenoiserModel(init=False).to(dev)
model.load_state_dict(torch.load(args.weights, map_location='cpu'))
model = add_flops_counting_methods(model)
model.eval().start_flops_count()

dataset = ExrDataset(want_reference=False,
                     training_path=args.inputs,
                     num_imgs=args.num_imgs,
                     cropsize=(args.img_height, args.img_width),
                     augment=False)

color, normal, albedo = dataset[0]
color, normal, albedo = color.to(dev), normal.to(dev), albedo.to(dev)
color, normal, albedo = pad_data(color), pad_data(normal), pad_data(albedo)
color, normal, albedo = color.unsqueeze(dim=0), normal.unsqueeze(dim=0), albedo.unsqueeze(dim=0)

with torch.no_grad():
    out = model(color, normal, albedo)

print('Output shape: {}'.format(list(out.shape)))
print('Flops:  {}'.format(flops_to_string(model.compute_average_flops_cost())))
print('Params: ' + get_model_parameters_number(model))
