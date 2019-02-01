import torch
import torch.cuda
import torch.optim
import signal
import sys
from torch.utils.data import DataLoader
from arg_handler import parse_train_args
from modified_model import DenoiserModel, TemporalDenoiserModel
from modified_vanilla_model import TemporalVanillaDenoiserModel, VanillaDenoiserModel
from dataset import NumpyRawDataset, PreProcessedDataset
from state import StateManager
from loss import Loss
from utils import iter_with_device
from cyclic import CyclicLR

args = parse_train_args()

dev = torch.device("cuda:{}".format(args.gpu))
if args.vanilla_net:
    model = TemporalVanillaDenoiserModel(init=args.restore is None).to(dev)
else:
    model = TemporalDenoiserModel(recurrent=not args.disable_recurrence, init=args.restore is None).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))#, weight_decay=1e-5)
state_mgr = StateManager(args, model, optimizer, dev)
loss_gen = Loss(dev)

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
scheduler = CyclicLR(optimizer, args.lr / 10, args.lr, step_size=10*num_batches)

val_dataset = PreProcessedDataset(dataset_path=args.validation_set, augment=False)
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=2,
                            shuffle=False, pin_memory=True)

def train_epoch(model, optimizer, scheduler, dataloader):
    #scheduler.step()
    model.train()
    for color, normal, albedo, ref in iter_with_device(dataloader, args.gpu):
        scheduler.batch_step()
        color, normal, albedo, ref = color.to(dev), normal.to(dev), albedo.to(dev), ref.to(dev)

        optimizer.zero_grad()
        outputs, e_irradiances = model(color, normal, albedo)
        loss, _ = loss_gen.compute(outputs, ref, color, albedo, e_irradiances)
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

    model.eval()
    total_val_loss = 0
    num_val_batches = 0
    for color, normal, albedo, ref in iter_with_device(val_dataloader, args.gpu):
        color, normal, albedo, ref = color.to(dev), normal.to(dev), albedo.to(dev), ref.to(dev)
        num_val_batches += 1
        with torch.no_grad():
            out, ei = model(color, normal, albedo)
            loss, _ = loss_gen.compute(out, ref, color, albedo, ei)
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
