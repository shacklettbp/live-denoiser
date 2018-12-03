import torch
import torchvision
from dataset import ExrDataset
from state import StateManager
from arg_handler import parse_infer_args
from model import DenoiserModel
from utils import tonemap
from data_loading import pad_data, save_exr
import os

args = parse_infer_args()

model = DenoiserModel(init=False).cuda()
model.load_state_dict(torch.load(args.weights, map_location='cpu'))

dataset = ExrDataset(training_path=args.inputs,
                     reference_path=args.inputs,
                     num_imgs=args.num_imgs,
                     cropsize=(args.img_height, args.img_width),
                     augment=False)

for i in range(args.start_frame, len(dataset)):
    color, _, normal, albedo = dataset[i]
    color, normal, albedo = color.cuda(), normal.cuda(), albedo.cuda()
    color, normal, albedo = pad_data(color), pad_data(normal), pad_data(albedo)
    color, normal, albedo = color.unsqueeze(dim=0), normal.unsqueeze(dim=0), albedo.unsqueeze(dim=0)

    with torch.no_grad():
        output = model(color, normal, albedo)
        output = output[..., 0:args.img_height, 0:args.img_width].cpu()
        output = output.squeeze()

    save_exr(output, os.path.join(args.outputs, 'out_{}.exr'.format(i)))

    tonemapped = tonemap(output)
    pil_txfm = torchvision.transforms.ToPILImage()
    img = pil_txfm(tonemapped)
    img.save(os.path.join(args.outputs, 'out_{}.png'.format(i)))
