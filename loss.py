import torch

def compute_loss(out, ref, input):
    err = (out - ref)**2/(out.detach()**2 + 0.01)

    loss = err.mean()
    return loss
