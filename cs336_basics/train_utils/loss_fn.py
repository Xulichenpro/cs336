import torch

def cross_entropy(
    input:torch.Tensor, #batch * vocab
    target:torch.Tensor,#batch
) -> torch.Tensor:
    max_input,indices = input.max(dim = -1,keepdim=True)
    suboutput = torch.exp(input - max_input).sum(dim = -1,keepdim=False)
    suboutput = torch.log(suboutput)
    target = target.unsqueeze(dim = -1)
    
    suboutput = (input.gather(dim = -1, index = target) - input.gather(dim = -1, index = indices)).squeeze(dim = -1) - suboutput
    output = -suboutput.mean() 
 
    return output


