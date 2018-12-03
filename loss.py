import torch

def compute_loss(out, ref):
    err = (out - ref)**2/(out.detach() + 0.01)**2

    loss = err.mean()
    return loss
