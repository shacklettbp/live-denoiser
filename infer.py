import torch
import torchvision
from torch.utils.data import DataLoader
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
import asyncio
import concurrent

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

dataloader = DataLoader(dataset, batch_size=1, num_workers=4,
                        shuffle=False,
                        pin_memory=True)

def save_results(out, albedo, ei, i):
    save_exr(out, os.path.join(args.outputs, 'out_{}.exr'.format(i)))
    #save_exr(albedo, os.path.join(args.outputs, 'albedo_out_{}.exr'.format(i)))
    #save_exr(ei, os.path.join(args.outputs, 'ei_{}.exr'.format(i)))

async def main():
    color_prev1 = torch.zeros_like(pad_data(dataset[0][0].to(dev)).unsqueeze(dim=0))
    color_prev2 = torch.zeros_like(color_prev1)

    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for idx, (color, normal, albedo) in enumerate(dataloader):
            color, normal, albedo = color.to(dev), normal.to(dev), albedo.to(dev)
            color, normal, albedo = pad_data(color), pad_data(normal), pad_data(albedo)
        
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
        
            await loop.run_in_executor(pool, save_results, output.cpu(), albedo_output.cpu(), e_irradiance.cpu(), idx)

asyncio.run(main())
