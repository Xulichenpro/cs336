import math
import torch
import torch.nn as nn

from einops import einsum,rearrange

from .rope_block import RoPE
from .linear_block import Linear

def softmax(x:torch.Tensor,dim:int) -> torch.Tensor:
    max_mat,_ = torch.max(x,dim=dim,keepdim=True)
    return torch.exp(x - max_mat) / torch.sum(torch.exp(x - max_mat),dim=dim,keepdim=True)

def scaled_dot_product_attention(
    q:torch.Tensor,
    k:torch.Tensor,
    v:torch.tensor,
    masked:torch.Tensor = None,
) -> torch.Tensor:
    d_k = q.shape[-1]
    pre_soft = einsum(
        q,
        k,
        "... seq_1 d_k,... seq_2 d_k -> ... seq_1 seq_2",
    )
    pre_soft = pre_soft / math.sqrt(d_k)
    
    # **Note**
    if masked is not None:
    # 假设 masked 是布尔类型（True表示保留，False表示遮蔽）
        pre_soft = pre_soft.masked_fill(masked == 0, float('-inf'))
    
    soft = softmax(pre_soft,-1)        
    return einsum(
        soft,
        v,
        "... seq_len1 seq_len2,... seq_len2 d_v ->... seq_len1 d_v"
    )

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model:int,
        num_heads:int,
        device:torch.device = None
    ):
        super().__init__()

        self.device = device if device else 'cpu'

        self.d_model = d_model
        self.h = num_heads
        self.d_k = self.d_model // self.h
        self.d_v = self.d_k
        self.WO = Linear(in_features=self.d_model,out_features=(self.h * self.d_k),device = self.device)
        self.Q = Linear(in_features=self.d_model,out_features=(self.h * self.d_k),device = self.device)
        self.K = Linear(in_features=self.d_model,out_features=(self.h * self.d_k),device = self.device)
        self.V = Linear(in_features=self.d_model,out_features=(self.h * self.d_v),device = self.device)

    def forward(
        self,
        x:torch.Tensor,
        rope:RoPE = None,
        token_positions:torch.Tensor = None,
    ) -> torch.Tensor:
        x = rearrange(
            x,
            "batch_size seq_len d_model -> 1 batch_size seq_len d_model",
        )

        # **Note**
        t = torch.arange(x.shape[-2], device=self.device).float().unsqueeze(dim = 0).unsqueeze(dim = 0)
        t = token_positions if token_positions is not None else t
        
        # **Note**
        wq = rearrange(
            self.Q.weight,
            "(h d_k) d_model -> h 1 d_k d_model",
            h = self.h,
            d_k = self.d_k,
        )
        wk = rearrange(
            self.K.weight,
            "(h d_k) d_model -> h 1 d_k d_model",
            h = self.h,
            d_k = self.d_k,
        )
        wv = rearrange(
            self.V.weight,
            "(h d_v) d_model -> h 1 d_v d_model",
            h = self.h,
            d_v = self.d_v,
        )

        Q = einsum(          
            x,
            wq,
            "one batch_size seq_len d_model,h one d_k d_model -> h batch_size seq_len d_k",
        )
        K = einsum(          
            x,
            wk,
            "one batch_size seq_len d_model,h one d_k d_model -> h batch_size seq_len d_k",
        )
        V = einsum(          
            x,
            wv,
            "one batch_size seq_len d_model,h one d_v d_model -> h batch_size seq_len d_v",
        )        

        if rope is not None:
            Q = rope.forward(Q,t)
            K = rope.forward(K,t)

        mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2],device=self.device))
        mask = mask.to(torch.bool)

        attention = scaled_dot_product_attention(Q,K,V,masked=mask)
        attention = rearrange(
            attention,
            "h batch_size seq_len d_v -> batch_size seq_len (h d_v)"
        )
        return self.WO(attention)



