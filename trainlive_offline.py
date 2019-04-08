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
                     training=True,
                     num_imgs=args.num_imgs)

for i in range(args.start_frame, len(dataset)):
    color, normal, albedo, reference, direct, indirect, tshadow = dataset[i]
    color, normal, albedo, reference = color.to(dev), normal.to(dev), albedo.to(dev), reference.to(dev)
    color, normal, albedo, reference = pad_data(color), pad_data(normal), pad_data(albedo), pad_data(reference)
    color, normal, albedo, reference = color.unsqueeze(dim=0), normal.unsqueeze(dim=0), albedo.unsqueeze(dim=0), reference.unsqueeze(dim=0)

    output = train_and_eval(training_state, color, reference, normal, albedo, False)
    output = output[0]

    save_exr(output, os.path.join(args.outputs, 'out_{}.exr'.format(i)))
