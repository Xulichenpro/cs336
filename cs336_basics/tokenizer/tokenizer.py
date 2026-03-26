import regex as re

from typing import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor,as_completed

from .bpe import updated_stats, merge

LEVEL = 255

class Tokenizer:
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple],special_tokens:list[str] = None,max_workers = 10):
        self.token2bytes = vocab
        self.bytes2token = {v:k for k,v in self.token2bytes.items()}
        self.merges = merges
        self.pair2token = {
            (self.bytes2token[px],self.bytes2token[py]):self.bytes2token[px + py] 
            for px,py in self.merges
        }
        self.special_tokens = special_tokens or []
        self.token2special = {}
        for byte,token in self.bytes2token.items():
            if byte.decode("utf-8",errors="replace") in self.special_tokens:
                self.token2special[token] = byte
        self.special2token = {v:k for k,v in self.token2special.items()}
        
        self.max_workers = max_workers

    def encode(self,text:str) -> list[int]:
        texts = self._split_by_special_keep(text)
        print(texts)
        bytes_stream = {}
        res = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.single_encode,text):id for id,text in enumerate(texts)
            }

            for future in as_completed(futures):
                id = futures[future]
                new_bytes = future.result()
                bytes_stream[id] = new_bytes
        for id in range(len(texts)):
            res.extend(bytes_stream[id])
        return res
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def single_encode(self,text:str) -> list[int]:
        if text == "":
            return []
        if text in self.special_tokens:
            return [self.special2token[text.encode('utf-8',errors='replace')]]
        
        bytes_stream = []
        for ch in text:
            bytes_stream.append(self.bytes2token[ch.encode()])

        while True:
            stats = {}
            stats = updated_stats(stats,bytes_stream)
            merged_pair = min(
                self.pair2token,
                key = lambda pair : stats.get(pair,float("inf"))
            )
            if stats.get(merged_pair,float("inf")) == float("inf") : break
            bytes_stream = merge(bytes_stream,merged_pair,self.pair2token[merged_pair])
            
        return bytes_stream
    
    def decode(self,bytes_stream:list[int]) -> str:
        utf_stream = []
        for byte in bytes_stream:
            if byte in self.token2special:
                utf_stream.extend(list(self.token2special[byte]))          
            else:
                utf_stream.extend(list(self.token2bytes[byte]))
        return bytes(utf_stream).decode("utf-8",errors="replace")
    
    def _split_by_special_keep(self,text):
        if not self.special_tokens:
            return [text]
        pattern = '(' + '|'.join(re.escape(s) for s in self.special_tokens) + ')'
        return re.split(pattern, text)


def main():
    text = "the cat ate<|endoftext|>the cat ate"
    print(text)
    vocb = {0: b' ',1: b'a',2:b'c', 3: b'e', 4: b'h',5: b't',6: b'th',7: b' c',8: b' a',9: b'the',10: b' at',11:b'<|endoftext|>'}
    merges = [(b't',b'h'),(b' ',b'c'),(b' ',b'a'),(b'th',b'e'),(b' a',b't')]
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer(vocb,merges,special_tokens)
    token_ids = tokenizer.encode(text)
    print(token_ids)
    text = tokenizer.decode(token_ids)
    print(text)

if __name__ == "__main__":
    main()
