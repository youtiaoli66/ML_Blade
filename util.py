
import torch
import random

def save_checkpoint(path, model, optimizer=None, scheduler=None):
    if optimizer is not None:
        optim_dict = optimizer.state_dict()
    else:
        optim_dict = None
    if scheduler:
        scheduler_state = scheduler.state_dict()
    else:
        scheduler_state = None
    torch.save(
        {
            'model': model.state_dict(),
            'optim': optim_dict,
            'scheduler': scheduler_state
        }, path
    )

def seed_setup(seed):
    torch.manual_seed(seed)
    # torch.manual_seed() 是设置随机种子，可以在多次运行中保持结果一致
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_params(model):
    count = 0
    for p in model.parameters():
        count += p.numel()
    return count
