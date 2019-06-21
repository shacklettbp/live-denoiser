import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import ExrDataset, ExrSeparatedDataset
from state import StateManager
from arg_handler import parse_live_infer_args
from modified_model import DenoiserModel
from modified_vanilla_model import VanillaDenoiserModelWrapper
from utils import tonemap
from data_loading import pad_data, save_exr, save_png
from filters import simple_filter
import os
import re
import asyncio
import concurrent
from trainlive import init_training_state, train_and_eval

args = parse_live_infer_args()
dev = torch.device('cuda:{}'.format(args.gpu))

training_state = init_training_state(dev, args.loss, args.frames_per_train, args.refsamples_per_train, args.iters_per_train, args.weights)

input_dir_base = os.path.normpath(args.inputs)

dataset = ExrSeparatedDataset(dataset_path=input_dir_base,
                              num_imgs=args.num_imgs, num_versions=16)

dataloader = DataLoader(dataset, batch_size=1, num_workers=4,
                        shuffle=False,
                        pin_memory=True)

if not os.path.isdir(args.outputs):
    os.makedirs(args.outputs, exist_ok = True)

def save_result(img, fname):
    save_exr(img, fname)

async def main():
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for i, (color, normal, albedo) in enumerate(dataloader):
            color, normal, albedo = color.to(dev), normal.to(dev), albedo.to(dev)
        
            output = train_and_eval(training_state, color, normal, albedo, False)
            output = output[0]
        
            await loop.run_in_executor(pool, save_result, output.cpu(), os.path.join(args.outputs, 'out_{}.exr'.format(i)))


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()

# asyncio.run(main())
