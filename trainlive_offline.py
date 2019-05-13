import torch
import torchvision
from dataset import ExrDataset
from state import StateManager
from arg_handler import parse_infer_args
from modified_model import DenoiserModel
from modified_vanilla_model import VanillaDenoiserModelWrapper
from utils import tonemap
from data_loading import pad_data, save_exr, save_png
from filters import simple_filter
import os
from trainlive import init_training_state, train_and_eval

args = parse_infer_args()
dev = torch.device('cuda:{}'.format(args.gpu))

training_state = init_training_state(dev, args.weights)

dataset = ExrDataset(dataset_path=args.inputs,
                     num_imgs=args.num_imgs)

alt_dataset = ExrDataset(dataset_path="/home/bps/rendering/data-fast/finals/1spp/bistro/inferpath_alt2",
                         num_imgs=args.num_imgs)

alt_dataset2 = ExrDataset(dataset_path="/home/bps/rendering/data-fast/finals/1spp/bistro/inferpath_alt3",
                         num_imgs=args.num_imgs)

alt_dataset3 = ExrDataset(dataset_path="/home/bps/rendering/data-fast/finals/1spp/bistro/inferpath_alt4",
                         num_imgs=args.num_imgs)

for i in range(args.start_frame, min(len(dataset), len(alt_dataset))):
    color, normal, albedo = dataset[i]
    color, normal, albedo = color.to(dev), normal.to(dev), albedo.to(dev)
    color, normal, albedo  = color.unsqueeze(dim=0), normal.unsqueeze(dim=0), albedo.unsqueeze(dim=0)

    alt_color, _, alt_albedo = alt_dataset[i]
    alt_color, alt_albedo = alt_color.to(dev), alt_albedo.to(dev)
    alt_color, alt_albedo = alt_color.unsqueeze(dim=0), alt_albedo.unsqueeze(dim=0)

    alt_color2, _, alt_albedo2 = alt_dataset2[i]
    alt_color2, alt_albedo2 = alt_color2.to(dev), alt_albedo2.to(dev)
    alt_color2, alt_albedo2 = alt_color2.unsqueeze(dim=0), alt_albedo2.unsqueeze(dim=0)

    alt_color3, _, alt_albedo3 = alt_dataset2[i]
    alt_color3, alt_albedo3 = alt_color3.to(dev), alt_albedo3.to(dev)
    alt_color3, alt_albedo3 = alt_color3.unsqueeze(dim=0), alt_albedo3.unsqueeze(dim=0)

    output = train_and_eval(training_state, color, normal, albedo, alt_color, alt_color2, alt_color3, alt_albedo, alt_albedo2, alt_albedo3, False)
    output = output[0]

    save_exr(output, os.path.join(args.outputs, 'out_{}.exr'.format(i)))
