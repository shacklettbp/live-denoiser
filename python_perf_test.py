import torch
import torchvision
from modified_vanilla_model import VanillaDenoiserModel
from modified_model import DenoiserModel
import sys
import time
import apex
from apex.fp16_utils import *

from data_loading import pad_data

torch.backends.cudnn.benchmark = True

model = VanillaDenoiserModel(init=True).cuda()
model = network_to_half(model)

random_input = torch.rand(1, 16, 1080, 1920)
random_input = pad_data(random_input).half().cuda()

train_input = torch.rand(2, 16, 256, 256).half().cuda()
train_ref = torch.rand(2, 3, 256, 256).half().cuda()

optimizer = FP16_Optimizer(apex.optimizers.FusedAdam(model.parameters(), 0.001),
                           dynamic_loss_scale=True)

model(train_input)
with torch.no_grad():
    model(random_input)

total = 0
for i in range(1000):
        torch.cuda.synchronize()
        t0 = time.time()

        optimizer.zero_grad()

        train_out, ei = model(train_input)
        loss = torch.abs(train_out - train_ref).mean()

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            out = model(random_input)
        torch.cuda.synchronize()
        end = time.time()
        total += end - t0

print(total / 1000)
