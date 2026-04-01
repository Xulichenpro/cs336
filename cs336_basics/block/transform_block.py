import torch
import torch.nn as nn

from .attention_block import MultiheadAttention
from .rms_norm_block import RMSNorm
from .swiglu_block import SWIglu
from .rope_block import RoPE

class Transformer(nn.Module):
    def __init__(
       self,
       d_model:int,
       num_heads:int,
       d_ff:int,
       device:torch.device = None, 
    ):
        super().__init__()

        self.device = device if device else 'cpu'

        self.d_model = d_model
        self.h = num_heads
        self.dff = d_ff

        self.multihead = MultiheadAttention(
            d_model=self.d_model,
            num_heads=self.h,
            device=self.device,
        )

        self.swiglu = SWIglu(
            d_model=self.d_model,
            dff=self.dff,
            device=self.device,
        )

        self.rms1 = RMSNorm(
            d_model=self.d_model,
            device=self.device,
        )
        self.rms2 = RMSNorm(
            d_model=self.d_model,
            device=self.device,
        )
    
    def forward(self,x:torch.Tensor,rope:RoPE = None) -> torch.Tensor:
        sub_out = x + self.multihead(x=self.rms1(x),rope=rope)
        output = sub_out + self.swiglu(x=self.rms2(sub_out))
        return output
