import time

import torch

from torch import nn
from functorch import vmap, jacrev, vjp
from functorch.compile import aot_function
from torch.autograd.functional import jacobian


def ts_compile(fx_g, inps):
  print("compiling")
  #f = torch.jit.script(fx_g)
  #f = torch.jit.freeze(f.eval())
  return fx_g


def ts_compiler(f):
  return aot_function(f, ts_compile)


def functorch_jacobian(points):
    """calculate a jacobian tensor along a batch of inputs. returns something of size
    `batch_size` x `output_dim` x `input_dim`"""
    return vmap(jacrev(model))(points)


def functorch_jacobian2(points):
    """calculate a jacobian tensor along a batch of inputs. returns something of size
    `batch_size` x `output_dim` x `input_dim`"""
    def _func_sum(points):
        return model(points).sum(dim=0)
    return jacrev(_func_sum)(points).permute(1, 0, 2)


def pytorch_jacobian(points):
    """calculate a jacobian tensor along a batch of inputs. returns something of size
    `batch_size` x `output_dim` x `input_dim`"""
    def _func_sum(points):
        return model(points).sum(dim=0)
    return jacobian(_func_sum, points, create_graph=True, vectorize=True).permute(1,0,2)

def fcn_scoring(f, name):
    import time
    iters=50
    for _ in range(5):
      f(points)
    begin = time.time()
    for _ in range(iters):
      f(points)
    print(name, (time.time()-begin)*1e6/iters)

torch.manual_seed(1234)


n_input, n_output, n_batch, n_hidden = 3, 5, 128, 64
print(n_input, n_output, n_batch, n_hidden)
model = nn.Sequential(nn.Linear(n_input, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output)).eval()
points = torch.rand((n_batch, n_input))

k = pytorch_jacobian(points)
v = functorch_jacobian(points)

# needed due to the global variables changing, otherwise it caches the function and doesn't recompute when the model changes.
functorch_jacobian3 = ts_compiler(lambda points: functorch_jacobian2(points))
m = functorch_jacobian3(points)
assert torch.allclose(k, v)
assert torch.allclose(k, m, rtol=1e-04)



fcn_scoring(pytorch_jacobian, name="pytorch")
fcn_scoring(functorch_jacobian, name="functorch1")
fcn_scoring(functorch_jacobian2, name="functorch2")
fcn_scoring(functorch_jacobian3, name="compiled functorch2")
# print()
