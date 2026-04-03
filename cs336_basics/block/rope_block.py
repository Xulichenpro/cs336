import math
import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(
        self,
        theta:float,
        d_k:int,
        max_seq_len:int,
        device:torch.device = None,
    ):
        super().__init__()

        self.device = device if device else 'cpu'
        self.theta = theta
        self.d = d_k
        self.max_seq_len = max_seq_len

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor):
        res = torch.zeros_like(x,device=self.device)

        for i in range(self.d // 2):
            with torch.no_grad():
                theta = token_positions / math.pow(self.theta,2 * i / self.d)
                sin = torch.sin(theta)
                cos = torch.cos(theta)
            res[...,2 * i] = cos * x[...,2 * i] - sin * x[...,2 * i + 1]
            res[...,2 * i + 1] = sin * x[...,2 * i] + cos * x[...,2 * i + 1]
        
        return res