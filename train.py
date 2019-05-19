import torch
import torch.cuda
import torch.optim
import signal
import sys
from torch.utils.data import DataLoader
from arg_handler import parse_train_args
from modified_model import DenoiserModel, TemporalDenoiserModel
from modified_vanilla_model import TemporalVanillaDenoiserModel, VanillaDenoiserModel
from smallmodel import TemporalSmallModel
from dataset import PreProcessedDataset
from state import StateManager
from loss import Loss
from utils import iter_with_device
from cyclic import CyclicLR
from filters import simple_filter
from data_loading import save_exr

args = parse_train_args()

dev = torch.device("cuda:{}".format(args.gpu))
if args.vanilla_net:
    model = TemporalSmallModel().to(dev)
    #model = TemporalVanillaDenoiserModel(init=True).to(dev)
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
#scheduler = CyclicLR(optimizer, args.lr / 10, args.lr, step_size=10*num_batches)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=10, patience=50)

val_dataset = PreProcessedDataset(dataset_path=args.validation_set, augment=False)
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=2,
                            shuffle=False, pin_memory=True)

def train_epoch(model, optimizer, scheduler, dataloader):
    #scheduler.step()
    model.train()
    for color, normal, albedo, ref, ref_albedo in iter_with_device(dataloader, args.gpu):
        #color, normal, albedo, ref, direct, indirect, tshadow = color.to(dev), normal.to(dev), albedo.to(dev), ref.to(dev), direct.to(dev), indirect.to(dev), tshadow.to(dev)
        color, normal, albedo, ref, ref_albedo = color.to(dev), normal.to(dev), albedo.to(dev), ref.to(dev), ref_albedo.to(dev)

        optimizer.zero_grad()
        outputs, e_irradiances, albedo_outs = model(color, normal, albedo)
        ref_e_irradiance = ref / (ref_albedo + 0.001)
        loss, _, _, _ = loss_gen.compute(ref, outputs, ref_e_irradiance, e_irradiances, ref_albedo, albedo_outs)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

    model.eval()
    total_val_loss = 0
    total_ei_loss = 0
    total_temporal_loss = 0
    total_albedo_loss = 0
    num_val_batches = 0
    for color, normal, albedo, ref, ref_albedo in iter_with_device(val_dataloader, args.gpu):
        #color, normal, albedo, ref, direct, indirect, tshadow = color.to(dev), normal.to(dev), albedo.to(dev), ref.to(dev), direct.to(dev), indirect.to(dev), tshadow.to(dev)
        color, normal, albedo, ref, ref_albedo = color.to(dev), normal.to(dev), albedo.to(dev), ref.to(dev), ref_albedo.to(dev)

        num_val_batches += 1
        with torch.no_grad():
            outputs, e_irradiances, albedo_outs = model(color, normal, albedo)
            ref_e_irradiance = ref / (ref_albedo + 0.001)
            loss, ei_loss, temp_loss, albedo_loss = loss_gen.compute(ref, outputs, ref_e_irradiance,  e_irradiances, ref_albedo, albedo_outs)
            total_val_loss += loss.cpu()
            total_ei_loss += ei_loss.cpu()
            total_temporal_loss += temp_loss.cpu()
            total_albedo_loss += albedo_loss.cpu()
    
    val_loss = total_val_loss / num_val_batches
    val_ei_loss = total_ei_loss / num_val_batches
    val_temp_loss = total_temporal_loss / num_val_batches
    val_albedo_loss = total_albedo_loss / num_val_batches
    print("Val loss: {}, EIrradiance Loss: {}, Temporal Loss: {}, Albedo Loss: {}".format(val_loss, val_ei_loss, val_temp_loss, val_albedo_loss))
    scheduler.step(val_loss)

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
