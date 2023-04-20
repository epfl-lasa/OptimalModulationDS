import torch
import torch.nn.functional as F
from functools import partial
from functorch import vmap, vjp
from functorch import jacrev
from functorch.compile import aot_function

import numpy as np

def fwd(x):
    x = x.unsqueeze(0)
    return x.T @ x

tensor_args = {'device': 'cpu', 'dtype': torch.float32}
input = torch.tensor([1, 3]).to(**tensor_args)

with torch.enable_grad():
    input.requires_grad = True
    input.grad = None

    output = fwd(input)

    m = torch.zeros((input.shape[0], output.shape[1]))
    m[0, 0] = 1
    m[1, 1] = 1

    print(m)
    output.backward(m)
    jac = input.grad.detach()

print(jac)
