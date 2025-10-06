import torch
import numpy as np

def precision(outputs, targets, denorm=None):
    if denorm is not None:
        return (1 - (torch.norm(denorm(outputs)-denorm(targets), p=2, dim=0) / torch.norm(denorm(targets), p=2, dim=0))) * 100
    else:
        return (1 - (torch.norm(outputs-targets, p=1, dim=0) / torch.norm(targets, p=1, dim=0))) * 100

def data_loss(outputs, targets, mask=None):
    if mask is None:
        return torch.nn.L1Loss()(outputs, targets)
    else:
        mask = mask.unsqueeze(1).repeat(1, 2, 1, 1)
        return torch.nn.L1Loss(reduction='none')(outputs, targets).flatten(-2)[mask.flatten(-2).bool()].mean()
