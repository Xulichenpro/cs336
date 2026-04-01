import torch
import numpy as np

def data_loader(
    dataset:np.ndarray,
    batch_size:int,
    context_length:int,
    device:str
) -> torch.Tensor:
    size = dataset.size
    indices = torch.randperm(size - context_length)[:batch_size]
    token_input  = torch.stack([torch.from_numpy(dataset[i : i + context_length])     for i in indices])
    token_target = torch.stack([torch.from_numpy(dataset[i + 1 : i + 1 + context_length]) for i in indices])

    return token_input.to(device), token_target.to(device)