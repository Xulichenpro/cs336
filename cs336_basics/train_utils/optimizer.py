import math
import torch

from collections.abc import Iterable

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        alpha: float = 1e-5,
        beta: tuple[float] = (0.9,0.999),
        epsilon: float = 1e-5,
        weight_decay: float = 1e-2,
    ):
        if alpha < 0:
            raise ValueError('The learning rate should be positive.')
        
        defaults = {
            "lr":alpha,
            "beta":beta,
            "epsilon":epsilon,
            "weight_decay":weight_decay,
        }

        super().__init__(params=params,defaults=defaults)
    
    @torch.no_grad()
    def step(self,closure = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            alpha = group['lr']
            beta1,beta2 = group['beta']
            epsilon = group['epsilon']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad
                m = state.get('m',torch.zeros_like(grad))
                v = state.get('v',torch.zeros_like(grad))
                t = state.get('t',1)

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                alpha_t = alpha * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= alpha_t * m / (torch.sqrt(v) + epsilon)
                p.data -= alpha * weight_decay * p.data

                state['m'] = m
                state['v'] = v
                state['t'] = t + 1
        
        return loss



def learning_rate_schedule(
    t:int,
    alpha_max:float,
    alpha_min:float,
    T_omiga:int,
    T_c:int,
) -> float:
    if t < T_omiga:
        lr = t * alpha_max / T_omiga

    elif t <= T_c:
        lr = alpha_min + 0.5 * (1 + math.cos((t - T_omiga) * math.pi / (T_c - T_omiga))) * (alpha_max - alpha_min)
    
    else:
        lr = alpha_min

    return lr

def grad_clipping(
    params:Iterable[torch.nn.Parameter],
    M:float,
    epsilon:float=1e-6,
):
    l2 = 0.0
    for p in params:
        if p.grad is None:
            continue     
        l2 += torch.sum(p.grad * p.grad).item()

    l2 = math.sqrt(l2)
    if l2 >= M:
        for p in params:
            if p.grad is None:
                continue
            p.grad *= M / (l2 + epsilon) 
