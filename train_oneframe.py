import torch
import torch.cuda
import torch.optim
import signal
import sys
import os
from torch.utils.data import DataLoader
from arg_handler import parse_train_args
from model import DenoiserModel, TemporalDenoiserModel
from vanilla_model import VanillaDenoiserModel
from dataset import NumpyRawDataset, PreProcessedDataset
from state import StateManager
from loss import compute_loss
from utils import iter_with_device
from cyclic import CyclicLR
from data_loading import pad_data, load_exr

args = parse_train_args()

dev = torch.device("cuda:{}".format(args.gpu))
if args.vanilla_net:
    model = VanillaDenoiserModel(init=args.restore is None).to(dev)
else:
    model = TemporalDenoiserModel(recurrent=not args.disable_recurrence, init=args.restore is None).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
state_mgr = StateManager(args, model, optimizer, dev)

#dataset = NumpyRawDataset((3, args.img_height, args.img_width),
#                          training_path=args.training_set,
#                          reference_path=args.reference_set,
#                          num_imgs=args.num_pairs)
dataset = PreProcessedDataset(dataset_path=args.training_set,
                              num_imgs=args.num_pairs,
                              augment=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8,
                        shuffle=True,
                        pin_memory=True)

num_batches = len(dataset) / args.batch_size

#def lr_schedule(epoch):
#    if epoch < 10:
#        return args.lr / 100
#    elif epoch < 20:
#        return args.lr * 1.65**(epoch - 10)/100
#
#    return args.lr
#
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
scheduler = CyclicLR(optimizer, args.lr / 10, args.lr, step_size=2*num_batches)

val_colors = []
val_normals = []
val_albedos = []
val_refs = []

for i in range(3):
    val_color = os.path.join(args.validation_set, 'hdr_{}.exr'.format(i))
    val_normal = os.path.join(args.validation_set, 'normal_{}.exr'.format(i))
    val_albedo = os.path.join(args.validation_set, 'albedo_{}.exr'.format(i))
    val_ref = os.path.join(args.validation_set, 'ref_{}.exr'.format(i))
    
    val_color = pad_data(load_exr(val_color))
    val_normal = pad_data(load_exr(val_normal))
    val_albedo = pad_data(load_exr(val_albedo))
    val_ref = pad_data(load_exr(val_ref))

    val_color[torch.isnan(val_color)] = 0
    val_ref[torch.isnan(val_ref)] = 0

    val_colors.append(val_color)
    val_normals.append(val_normal)
    val_albedos.append(val_albedo)
    val_refs.append(val_ref)

val_color = torch.stack(val_colors).unsqueeze(dim=0).to(dev)
val_normal = torch.stack(val_normals).unsqueeze(dim=0).to(dev)
val_albedo = torch.stack(val_albedos).unsqueeze(dim=0).to(dev)
val_ref = torch.stack(val_refs).unsqueeze(dim=0).to(dev)

def train_epoch(model, optimizer, scheduler, dataloader):
    optimizer.zero_grad()
    #scheduler.step()
    model.train()
    for color, normal, albedo, ref in iter_with_device(dataloader, args.gpu):
        scheduler.batch_step()
        color, normal, albedo, ref = color.to(dev), normal.to(dev), albedo.to(dev), ref.to(dev)

        outputs = model(color, normal, albedo)
        loss = compute_loss(outputs, ref, color)
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        val_out = model(val_color, val_normal, val_albedo)
        total_val_loss = compute_loss(val_out, val_ref, val_color)
        print("Val loss: {}".format(total_val_loss))

def train(model, optimizer, scheduler, dataloader, state_mgr, num_epochs):
    interrupted = False
    def handler(sig, fr):
        nonlocal interrupted
        if interrupted:
            sys.exit(1)
        interrupted = True
    signal.signal(signal.SIGINT, handler)

    model.train()
    for i in range(state_mgr.get_start_epoch(), num_epochs):
        print("Epoch {}".format(i + 1))

        train_epoch(model, optimizer, scheduler, dataloader)

        if (i + 1) % 10 == 0 or interrupted:
            state_mgr.save_state(model, optimizer, i + 1)

        if interrupted:
            break

train(model, optimizer, scheduler, dataloader, state_mgr, args.epochs)
