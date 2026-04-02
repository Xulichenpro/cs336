import os
import yaml
import torch
import logging
import time
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from einops import rearrange

from tokenizer.tokenizer import Tokenizer
from tokenizer.pretokenizer import find_chunk_boundaries
from block.rope_block import RoPE
from block.lm import TransformerLM
from train_utils.dataloader import data_loader
from train_utils.loss_fn import cross_entropy
from train_utils.optimizer import AdamW,grad_clipping,learning_rate_schedule
from train_utils.checkpoint_utils import save_checkpoint

DATA_DIR = Path(__file__).parent / "data"
TRAIN_FILE = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
TEST_FILE = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
TRAIN_CACHE_FILE = DATA_DIR / "train.bin"
TEST_CACHE_FILE = DATA_DIR / "test.bin"

MODEL_DIR = Path(__file__).parent / "model"
VOCAB_PATH = MODEL_DIR / "vocab.pkl"
MERGES_PATH = MODEL_DIR / "merges.pkl"
MODEL_DIR.mkdir(exist_ok=True,parents=True)

SPECIAL_TOKENS = ["<|endoftext|>"]

MAX_TOTAL_TOKENS = 327680000

def setup_logger(name: str) -> logging.Logger:
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件 handler：记录所有级别
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # 控制台 handler：只显示 INFO 及以上
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger

def draw_loss_curve(
    train_losses:list,
    val_losses:list
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 2. 绘制第一个子图：Train Loss
    ax1.plot(train_losses, color='blue', label='Train Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 3. 绘制第二个子图：Val Loss
    ax2.plot(val_losses, color='red', label='Val Loss')
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('t')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    # 4. 自动调整布局，防止子图之间的标签重叠
    plt.tight_layout()
    plt.show()

def lazy_load(
    file_path:Path,
    cache_path:Path,
    tokenizer:Tokenizer,
    logger: logging.Logger,
) -> np.memmap:
    if not cache_path.exists():
        logger.info(f"✏️  Tokenizing {file_path.name} → {cache_path.name} ...")
        
        tokens_len = 0
            
        with open(file_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f,16, b"<|endoftext|>")        
            # 关键修改：以追加二进制模式 ('ab') 先打开缓存文件
            with open(cache_path, 'ab') as cache_f: 
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    f.seek(start)
                    logger.info(f"⚙️  Reading from {file_path}...")
                    chunk = f.read(end - start).decode("utf-8", errors="ignore") 
                    chunks = tokenizer._split_by_special_keep(chunk)
           
                    for chunk in chunks: 
                        tokens = tokenizer.encode(chunk)   
                        tokens_len += len(tokens)      
                        tokens = np.array(tokens, dtype=np.uint16)                  
                            # 将文件句柄 cache_f 传给 tofile，实现追加
                        tokens.tofile(cache_f)                    
                          
                    logger.info(f"🧠 Have encoded {tokens_len} tokens")   
        # with open(file_path,'r') as f:
        #     text = f.read()
   
        logger.info(f"💾 Saved {tokens_len:,} tokens to {cache_path.name}")
    else:
        logger.info(f"✅ Cache found: {cache_path.name}")
    return np.memmap(cache_path, dtype=np.uint16, mode='r')

def main():
    logger = setup_logger("train")

    hyper_file = Path(__file__).parent / "hyperparam.yml"
    with open(hyper_file,'r') as f:
        hyper_params = yaml.safe_load(f)

    logger.info(f"📋 Loaded hyperparams from {hyper_file}")
    logger.debug(f"Hyperparams: {hyper_params}")  

    batch_size = hyper_params["batch_size"]
    context_length = hyper_params["context_length"]
    max_grad_norm = hyper_params["max_grad_norm"]
    eval_interval  = hyper_params["eval_interval"]
    eval_iters = hyper_params["eval_iters"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    total_steps = MAX_TOTAL_TOKENS // (batch_size * context_length)

    logger.info(f"💻 Device: {device}")
    logger.info(
        f"🚀 Total steps: {total_steps:,} "
        f"(batch={batch_size}, ctx={context_length}, tokens={MAX_TOTAL_TOKENS:,})"
    )
   
    tokenizer = Tokenizer.from_path(
        VOCAB_PATH,
        MERGES_PATH,
        special_tokens=SPECIAL_TOKENS,
        logger = logger,
        max_workers = 16,
    )
    lm = TransformerLM(
        device=device,
        **hyper_params["TransformerLM"],       
    ).to(device)
    optimizer = AdamW(
        params = lm.parameters(),
        **hyper_params["AdamW"]
    )
    rope = RoPE(
        device=device,
        **hyper_params["RoPE"]
    ).to(device)

    n_params = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    logger.info(f"🧠 Model parameters: {n_params:,}")

    train_data = lazy_load(TRAIN_FILE,TRAIN_CACHE_FILE,tokenizer,logger)
    test_data = lazy_load(TEST_FILE,TEST_CACHE_FILE,tokenizer,logger)

    logger.info("🏋️  Starting training ...")

    lm.train()
    t_start = time.time()
    t_step  = time.time()

    train_losses = []
    val_losses = []
   
    for step in range(total_steps):

        lr = learning_rate_schedule(step, **hyper_params["lr_schedule"])
        for group in optimizer.param_groups:
            group['lr'] = lr
        X, y = data_loader(train_data,batch_size,context_length,device)

        optimizer.zero_grad()
        pred = lm(X,rope)
        
        pred = rearrange(
            pred,
            "batch_size seq_len vocab_size -> (batch_size seq_len) vocab_size",
        )
        
        y = rearrange(
            y,
            "batch_size seq_len -> (batch_size seq_len)",
        )

        
        train_loss = cross_entropy(pred,y)
        train_loss.backward()
        grad_clipping(lm.parameters(),max_grad_norm)
        optimizer.step()

        train_losses.append(train_loss.item())
        step_time      = time.time() - t_step
        t_step         = time.time()
        tokens_per_sec = (batch_size * context_length) / max(step_time, 1e-9)
        elapsed        = time.time() - t_start

        logger.debug(
            f"⚙️  step={step:>6}/{total_steps} "
            f"train_loss={train_loss.item():.4f} "
            f"lr={lr:.2e} "
            f"tok/s={tokens_per_sec:,.0f} "
            f"elapsed={elapsed:.1f}s"
        )

        if step % eval_interval == 0:
            lm.eval()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(eval_iters):
                    X, y = data_loader(test_data, batch_size, context_length, device)
                    pred = lm.forward(X,rope)
                    pred = rearrange(
                        pred,
                        "batch_size seq_len vocab_size -> (batch_size seq_len) vocab_size",
                    )
                    y = rearrange(
                        y,
                        "batch_size seq_len -> (batch_size seq_len)",
                    )

                    val_loss += cross_entropy(pred,y).item()
                val_loss /= eval_iters
                val_losses.append(val_loss)
                ppl = torch.exp(torch.tensor(val_loss)).item()

                logger.info(
                    f"😎 step={step:>6}/{total_steps} "
                    f"train_loss={train_loss.item():.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"ppl={ppl:.2f} "
                    f"lr={lr:.2e} "
                    f"tok/s={tokens_per_sec:,.0f} "
                    f"elapsed={elapsed:.1f}s"
                )   
            lm.train()
        
        if step % 1000 == 0:
            draw_loss_curve(train_losses,val_losses)
    
    total_time = time.time() - t_start
    logger.info(
        f"🎉 Training complete! "
        f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}"
    )

if __name__ == "__main__":
    main()
 



        