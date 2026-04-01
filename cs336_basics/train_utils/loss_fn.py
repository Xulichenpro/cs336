import torch

def cross_entropy(
    input:torch.Tensor, #batch * vocab
    target:torch.Tensor,#batch
) -> torch.Tensor:
    max_input,indices = input.max(dim = -1)
    max_input = max_input.unsqueeze(dim = -1)
    suboutput = torch.exp(input - max_input).sum(dim = -1,keepdim=False)
    suboutput = torch.log(suboutput)

    suboutput = (input[torch.arange(input.shape[-2]),target] - input[torch.arange(input.shape[-2]),indices]) - suboutput
    output = suboutput.sum() / input.shape[-2]
    output.data = -output.data

    return output