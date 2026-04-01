import torch
import torch.nn as nn

from .embedding_block import Embedding
from .transform_block import Transformer
from .rms_norm_block import RMSNorm
from .rope_block import RoPE
from .linear_block import Linear

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size:int,
        context_length:int,
        num_layers:int,
        d_model:int,
        num_heads:int,
        d_ff:int,
        device:torch.device = None,
    ):
        super().__init__()

        self.device = device if device else 'cpu'

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.h = num_heads
        self.dff = d_ff
        
        self.embedding = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
            device = self.device,
        )

        self.trans_blocks = [
            Transformer(
                d_model=self.d_model,
                num_heads=self.h,
                d_ff=self.dff,
                device=self.device,
            )
            for _ in range(self.num_layers)
        ]

        self.final_rms = RMSNorm(
            d_model=self.d_model,
            device=self.device,
        )

        self.lm_head = Linear(
            in_features=self.d_model,
            out_features=self.vocab_size,
            device=self.device,
        )

    def forward(self,x:torch.Tensor,rope:RoPE = None) -> torch.Tensor:
        sub_output = self.embedding(x)

        for num_layer in range(self.num_layers):
            sub_output = self.trans_blocks[num_layer](sub_output,rope = rope)

        output = self.lm_head(
            self.final_rms(sub_output),
        )

        return output