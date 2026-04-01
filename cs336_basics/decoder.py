import torch
import torch.nn.functional as F

from block.lm import TransformerLM
from block.rope_block import RoPE
from block.attention_block import softmax

END = '<|endoftext|>'

def decoder(
    input:torch.Tensor,
    model:TransformerLM,
    vocab:dict[int,bytes],
    max_tokens:int,
    temperature:float,
    top_p:float,
    rope:RoPE = None,
):
    output = []
    for _ in range(max_tokens):
        input = input.unsqueeze(dim = 0)
        model_output = model.forward(input,rope=rope).squeeze(dim = 0) #seq_len d_model
        logits = model_output[-1] #d_model
        
        next_token = top_p_sampling(logits,p = top_p, temperature= temperature)
        input = input.squeeze(dim = 0)
        input = torch.cat([input,next_token],dim = -1)

        byte_stream = vocab[next_token.item()]
        next_str = byte_stream.decode('utf-8',errors='replace')
        output.append(next_str)
        if next_str == END:
            break
    return output


def top_p_sampling(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """
    Args:
        logits:      形状 (batch_size, vocab_size) 或 (vocab_size,)
        p:           核概率阈值，范围 (0, 1]
        temperature: 温度系数，越高分布越平滑，越低越尖锐

    Returns:
        sampled_token: 形状 (batch_size,) 或标量，采样到的 token id
    """
    # ── 统一处理成 2D ────────────────────────────────────────
    squeeze = logits.dim() == 1
    if squeeze:
        logits = logits.unsqueeze(0)          # (1, vocab_size)

    # ── Step 1：温度缩放 ─────────────────────────────────────
    logits = logits / temperature

    # ── Step 2：转为概率分布 ─────────────────────────────────
    probs = F.softmax(logits, dim=-1)         # (batch, vocab)

    # ── Step 3：按概率从高到低排序 ───────────────────────────
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # ── Step 4：计算累积概率 ─────────────────────────────────
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # (batch, vocab)

    # ── Step 5：生成截断掩码 ─────────────────────────────────
    # 将累积概率 > p 的位置置为 True（需要移除）
    # 为保留累积概率"刚好越过 p"的那个 token，向右移一位再做掩码
    remove_mask = cumulative_probs - sorted_probs > p      # (batch, vocab)

    # ── Step 6：将被截断 token 的概率置为 0 ──────────────────
    sorted_probs[remove_mask] = 0.0

    # ── Step 7：重归一化 ─────────────────────────────────────
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # ── Step 8：从过滤后的分布中采样 ────────────────────────
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)  # (batch, 1)

    # ── Step 9：映射回原始词表下标 ───────────────────────────
    sampled_token = sorted_indices.gather(dim=-1, index=sampled_sorted_idx)  # (batch, 1)
    sampled_token = sampled_token.squeeze(-1)  # (batch,)

    return sampled_token.squeeze(0) if squeeze else sampled_token