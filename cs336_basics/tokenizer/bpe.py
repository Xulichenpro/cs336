from pathlib import Path

def updated_stats(stats:dict[tuple,int],bytes_stream:list,cache:dict[tuple,dict] = None) -> dict[tuple,int]:
    for id,(px,py) in enumerate(zip(bytes_stream,bytes_stream[1:])):
        stats[(px,py)] = stats.get((px,py),0) + 1
    
    if cache is not None:
        cache[tuple(bytes_stream)] = cache.get(tuple(bytes_stream),0) + 1
    
    return stats,cache

def merge(bytes:list,merge_id:tuple,new_id:int) -> list:
    new_bytes = []

    i = 0
    while i < len(bytes):
        if i < len(bytes) - 1 and bytes[i] == merge_id[0] and bytes[i + 1] == merge_id[1] :
            new_bytes.append(new_id)
            i += 2
        else:
            new_bytes.append(bytes[i])
            i += 1
    
    return new_bytes

def main():
    text_path = Path(__file__).parent / "test.txt"
    with open(text_path,'r') as f:
        text = f.read()

    print(f"raw text:{text}")
    bytes = list(text.encode('utf-8'))
    print(f"byte stream:{bytes}")
    stats = updated_stats({},bytes)
    print(f"stats:{stats}")

if __name__ == "__main__":
    main()