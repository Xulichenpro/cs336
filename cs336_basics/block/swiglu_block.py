import torch
import torch.nn as nn

from .linear_block import Linear

class SWIglu(nn.Module):
    def __init__(
        self,
        d_model:int,
        dff:int = None,
        device:torch.device = None,
        dtype:torch.dtype = None,
    ):
        super().__init__()

        self.device = device if device else 'cpu'
        self.dtype = dtype if dtype else torch.float32
        self.d_model = d_model
        self.dff = dff if dff else 8 * self.d_model // 3

        self.linear_unit1 = Linear(
            self.d_model,
            self.dff,
            device = self.device,
            dtype = self.dtype,
        )
        self.linear_unit2 = Linear(
            self.dff,
            self.d_model,
            device = self.device,
            dtype = self.dtype
        )
        self.linear_unit3 = Linear(
            self.d_model,
            self.dff,
            device = self.device,
            dtype = self.dtype,
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        w1x = self.linear_unit1(x)
        return self.linear_unit2(
            self.linear_unit3(x) * (w1x * torch.sigmoid(w1x))
        )