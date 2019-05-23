import torch
import torchvision
from dataset import ExrDataset, ExrSeparatedDataset
from state import StateManager
from arg_handler import parse_infer_args
from modified_model import DenoiserModel
from modified_vanilla_model import VanillaDenoiserModelWrapper
from utils import tonemap
from data_loading import pad_data, save_exr, save_png
from filters import simple_filter
import os
import re
from trainlive import init_training_state, train_and_eval

args = parse_infer_args()
dev = torch.device('cuda:{}'.format(args.gpu))

training_state = init_training_state(dev, args.weights)

input_dir_base = os.path.normpath(args.inputs)

dataset = ExrSeparatedDataset(dataset_path=input_dir_base,
                              num_imgs=args.num_imgs, num_versions=5)

for i in range(args.start_frame, len(dataset)):
    color, normal, albedo = dataset[i]
    color, normal, albedo = color.to(dev), normal.to(dev), albedo.to(dev)
    color, normal, albedo  = color.unsqueeze(dim=0), normal.unsqueeze(dim=0), albedo.unsqueeze(dim=0)

    output = train_and_eval(training_state, color, normal, albedo, False)
    output = output[0]

    save_exr(output, os.path.join(args.outputs, 'out_{}.exr'.format(i)))
