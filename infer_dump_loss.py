import os
import sys
import torch
import torchvision
from dataset import ExrDataset
from state import StateManager
from arg_handler import parse_infer_args
from model import TemporalDenoiserModel
from vanilla_model import TemporalVanillaDenoiserModel
from utils import tonemap
from data_loading import pad_data, save_exr, load_exr
from loss import Loss

args = parse_infer_args()
dev = torch.device('cuda:{}'.format(args.gpu))
if args.vanilla_net:
    model = TemporalVanillaDenoiserModel(init=False).to(dev)
else:
    model = TemporalDenoiserModel(recurrent=not args.disable_recurrence, init=False).to(dev)

model.load_state_dict(torch.load(args.weights, map_location='cpu'))
model.eval()

dataset = ExrDataset(want_reference=False,
                     training_path=args.inputs,
                     num_imgs=args.num_imgs,
                     cropsize=(args.img_height, args.img_width),
                     augment=False)

if args.loss_check is None:
    print("Provide reference image", file=sys.stderr)
    sys.exit(1)

ref_frame = load_exr(args.loss_check)

loss_gen = Loss(dev)

for i in range(args.start_frame, len(dataset)):
    color, normal, albedo = dataset[i]
    color, normal, albedo = color.to(dev), normal.to(dev), albedo.to(dev)
    color[torch.isnan(color)] = 0
    color, normal, albedo = pad_data(color), pad_data(normal), pad_data(albedo)
    color, normal, albedo = color.unsqueeze(dim=0), normal.unsqueeze(dim=0), albedo.unsqueeze(dim=0)

    color_prev1 = torch.zeros_like(color)
    color_prev2 = torch.zeros_like(color)

    with torch.no_grad():
        output = model(color.unsqueeze(dim=1), normal.unsqueeze(dim=1), albedo.unsqueeze(dim=1),
                       color_prev1=color_prev1, color_prev2=color_prev2)

        output = output[..., 0:args.img_height, 0:args.img_width].cpu()
        output = output.squeeze()

        if args.disable_recurrence:
            color_prev1 = color
        else:
            color_prev1 = output

        color_prev2 = color_prev1

        _, loss = loss_gen.compute(output.unsqueeze(dim=0), ref_frame.unsqueeze(dim=0), color.unsqueeze(dim=0))

    save_exr(loss, os.path.join(args.outputs, 'loss_{}.exr'.format(i)))

    tonemapped = tonemap(loss)
    pil_txfm = torchvision.transforms.ToPILImage()
    tonemapped = tonemapped.clamp(0.0, 1.0)
    img = pil_txfm(tonemapped)
    img.save(os.path.join(args.outputs, 'loss_{}.png'.format(i)))
