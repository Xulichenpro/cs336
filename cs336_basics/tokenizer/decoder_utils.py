LEVEL = 255

def bytes_decoder(token2pair:dict[int,tuple],token2bytes:dict[int,list],speical_tokens:dict[str,int]) -> dict[int,list] :
    if speical_tokens:
        max_id = speical_tokens[min(speical_tokens,key=lambda k : speical_tokens[k])]
    else:
        max_id = max(token2pair) + 1
    
    pre_max_id = max(token2bytes) 
    for key,(px,py) in token2pair.items(): 
        if key <= pre_max_id or key >= max_id: continue      
        token2bytes[key] = token2bytes[px] + token2bytes[py]
       
    return token2bytes

def main():
    token2pair = {
        256: (48,49),
        257: (49,50),
        258: (256,23),
        259: (24,257),
        260: (258,259),
    }
    token2bytes = {num:[num] for num in range(LEVEL + 1)}

    token2bytes = bytes_decoder(token2pair,token2bytes)
    print(token2bytes)
    token2pair[261] = (50,51)
    token2bytes = bytes_decoder(token2pair,token2bytes)
    print(token2bytes)

if __name__ == "__main__":
    main()    