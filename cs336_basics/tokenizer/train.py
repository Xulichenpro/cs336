import os
import regex as re
import multiprocessing as mp

from pathlib import Path
from multiprocessing import Pool

from .bpe import updated_stats,merge
from .serialize_bpe import save_pkl
from .pretokenizer import find_chunk_boundaries

FILE_PATH = Path(__file__).parent.parent / "data/TinyStoriesV2-GPT4-train.txt"
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
LEVEL = 255

def train_bpe(
        input_path:str,
        vocab_size:int,
        special_tokens:list[str],
        num_processes = 16,
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    assert vocab_size >= LEVEL + 1,"unvalid vocab size"

    special_tokens.sort(key=len, reverse=True)

    chunks = []
    global_stats = {}
    global_cache = {}
    token2bytes = {num:num.to_bytes(1,'big') for num in range(LEVEL + 1)}
    bytes2token = {v:k for k,v in token2bytes.items()}
    merges = []

    with open(input_path,'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    stats_cache_list = []
    start = 0
    ctx = mp.get_context("spawn")
    
    with ctx.Pool(num_processes) as pool:
        new_stats_cache_list = pool.starmap(
            pretokenize,
            [(chunk,special_tokens) for chunk in chunks]
        )
        stats_cache_list.extend(new_stats_cache_list)

    for stats,cache in stats_cache_list:
        for pair,cnt in stats.items():
            global_stats[pair] = global_stats.get(pair,0) + cnt
        for bytes_stream,cnt in cache.items():
            global_cache[bytes_stream] = global_cache.get(bytes_stream,0) + cnt

    for token_id in range(LEVEL + 1,vocab_size - len(special_tokens)):
        if not global_stats:
            break

        merged_pair = max(
            global_stats,
            key = lambda pair : (
                global_stats[pair],
                (token2bytes[pair[0]]),
                (token2bytes[pair[1]]),
            )
        )
        new_bytes = token2bytes[merged_pair[0]] + token2bytes[merged_pair[1]]
        token2bytes[token_id] = new_bytes
        bytes2token[new_bytes] = token_id
        merges.append((token2bytes[merged_pair[0]],token2bytes[merged_pair[1]]))
        
        iter = list(global_cache.keys())
        
        for bytes_stream in iter:
            if merged_pair not in zip(list(bytes_stream),list(bytes_stream)[1:]) : 
                continue

            cnt = global_cache[bytes_stream]
            old_bytes_stream = list(bytes_stream)
            bytes_stream = merge(old_bytes_stream,merged_pair,token_id)
            
            for i,id in enumerate(bytes_stream):
                if token_id != id : continue
                if i >= 1 :                 
                    global_stats[(bytes_stream[i - 1],token_id)] = cnt + global_stats.get((bytes_stream[i - 1],token_id),0)
                    if bytes_stream[i - 1] != token_id:                      
                        global_stats[(bytes_stream[i - 1],merged_pair[0])] = global_stats[(bytes_stream[i - 1],merged_pair[0])] - cnt
                        if global_stats[(bytes_stream[i - 1],merged_pair[0])] == 0:
                            del global_stats[(bytes_stream[i - 1],merged_pair[0])]
                    else:
                        global_stats[(merged_pair[1],merged_pair[0])] -= cnt
                        if global_stats[(merged_pair[1],merged_pair[0])] == 0:
                            del global_stats[(merged_pair[1],merged_pair[0])]
                    
                if i <= len(bytes_stream) - 2:                
                    if bytes_stream[i + 1] != token_id:
                        global_stats[(token_id,bytes_stream[i + 1])] = cnt + global_stats.get((token_id,bytes_stream[i + 1]),0)
                        global_stats[(merged_pair[1],bytes_stream[i + 1])] -= cnt
                        if global_stats[(merged_pair[1],bytes_stream[i + 1])] == 0:
                            del global_stats[(merged_pair[1],bytes_stream[i + 1])] 
        
            global_cache[tuple(bytes_stream)] = global_cache.get(tuple(bytes_stream),0) + cnt
            del global_cache[tuple(old_bytes_stream)]

        del global_stats[merged_pair]
              
    for id,token_id in enumerate(range(len(token2bytes),len(token2bytes) + len(special_tokens))):
        token2bytes[token_id] = special_tokens[id].encode("utf-8")

    return token2bytes,merges

def pretokenize(text:str,special_tokens = SPECIAL_TOKENS):
    texts = _split_by_special_tokens(text,special_tokens=special_tokens)
    stats = {}
    cache = {}
 
    for piece in texts:
        s, c = single_pretokenize(piece)
        for pair, cnt in s.items():
            stats[pair] = stats.get(pair, 0) + cnt
        for bs, cnt in c.items():
            cache[bs] = cache.get(bs, 0) + cnt
    return stats, cache

def single_pretokenize(chunk:str):
    stats = {}
    cache = {}
    match_iter =  re.finditer(PAT,chunk)
    for match in match_iter:
        stats,cache = updated_stats(stats,list(str(match.group()).encode('utf-8')),cache=cache)
    return stats,cache

def _split_by_special_tokens(text:str,special_tokens:list[str] = None) -> list[str]:
    if not special_tokens:
        return [text]
    pattern = '|'.join(re.escape(s) for s in special_tokens) 
    texts = re.split(pattern, text)
    texts = [t for t in texts if t]
    return texts

def main():
    model_dir = Path(__file__).parent.parent / "model"
    model_dir.mkdir(parents=True,exist_ok=True)
    token2bytes,merges = train_bpe(FILE_PATH,VOCAB_SIZE,SPECIAL_TOKENS)

    print(token2bytes,merges,sep='\n')
    
    save_pkl(token2bytes,'vocab',model_dir / "vocab.pkl")
    save_pkl(merges,'merges', model_dir / 'merges.pkl')

if __name__ == "__main__":
    mp.freeze_support()
    main()