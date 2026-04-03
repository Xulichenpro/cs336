import torch
import numpy as np

def data_loader(
    dataset:np.memmap,
    batch_size:int,
    context_length:int,
    device:str
) -> torch.Tensor:
    size = dataset.size
    print(f"DEBUG: dataset type: {type(dataset)}, shape: {getattr(dataset, 'shape', 'No Shape')}")

    indices = torch.randint(0,size - context_length, (batch_size,))
    token_input  = torch.stack([torch.tensor(dataset[i : i + context_length],dtype=torch.long)     for i in indices])
    token_target = torch.stack([torch.tensor(dataset[i + 1 : i + 1 + context_length], dtype=torch.long) for i in indices])

    return token_input.to(device), token_target.to(device)