import torch
import torch.cuda
import torch.optim
import signal
import sys
from torch.utils.data import DataLoader
from arg_handler import parse_train_args
from model import DenoiserModel
from dataset import NumpyRawDataset, PreProcessedDataset
from state import StateManager
from loss import compute_loss
from utils import iter_with_device
from cyclic import CyclicLR

args = parse_train_args()

dev = torch.device("cuda:{}".format(args.gpu))
model = DenoiserModel(init=args.restore is None).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
state_mgr = StateManager(args, model, optimizer, dev)

#dataset = NumpyRawDataset((3, args.img_height, args.img_width),
#                          training_path=args.training_set,
#                          reference_path=args.reference_set,
#                          num_imgs=args.num_pairs)
dataset = PreProcessedDataset(dataset_path=args.training_set,
                              num_imgs=args.num_pairs)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8,
                        shuffle=True,
                        pin_memory=True)

num_batches = len(dataset) / args.batch_size

scheduler = CyclicLR(optimizer, args.lr / 10, args.lr, step_size=2*num_batches)

val_dataset = PreProcessedDataset(dataset_path=args.validation_set)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8,
                            shuffle=False, pin_memory=True)

def train_epoch(model, optimizer, scheduler, dataloader):
    model.train()
    for color, normal, albedo, ref in iter_with_device(dataloader, args.gpu):
        scheduler.batch_step()
        color, normal, albedo, ref = color.to(dev), normal.to(dev), albedo.to(dev), ref.to(dev)

        optimizer.zero_grad()
        output = model(color, normal, albedo)
        loss = compute_loss(output, ref)
        loss.backward()
        optimizer.step()

    model.eval()
    total_val_loss = 0
    num_val_batches = 0
    for color, normal, albedo, ref in iter_with_device(dataloader, args.gpu):
        color, normal, albedo, ref = color.to(dev), normal.to(dev), albedo.to(dev), ref.to(dev)
        num_val_batches += 1
        with torch.no_grad():
            out = model(color, normal, albedo)
            loss = compute_loss(out, ref)
            total_val_loss += loss.cpu()
    
    print("Val loss: {}".format(total_val_loss / num_val_batches))

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
