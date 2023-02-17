import torch, time
torch.random.manual_seed(1337)
@torch.jit.script
def fn1(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    for i in range(n):
        tmp = x[i]
        x[i][tmp < i] = 0
    return x
@torch.jit.script
def fn2(x: torch.Tensor) -> torch.Tensor:
    for i, row in enumerate(x):
        for j, element in enumerate(row):
            if element in x[element.int()]:
                x[i, j] = 0
    return x

sz = 1000
n_k = 100
x = (sz*torch.rand(sz, n_k)).int()
a = fn2(x)
print('Done!')
for i in range(3):
    y = fn2(x)

t0 = time.time()
N_REP = 2
for i in range(N_REP):
    y = fn2(x)
t1 = time.time()
print("Time: %4.2fms" % (1000*(t1-t0)/N_REP))
#
# x = torch.tensor([[2, 4], [3, 6], [2, 4], [1, 3], [2, 3], [3, 4]])