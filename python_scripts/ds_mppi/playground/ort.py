import torch

dof = 3

def ort(vec):
    res = torch.zeros(vec.shape[0], vec.shape[0])
    res[:, 0] = vec.squeeze()
    for i in range(1, vec.shape[0]):
        col = res[:, i]
        for j in range(vec.shape[0]):
            if j == 0:
                col[j] = -vec[i, 0]
            elif j == i:
                col[j] = vec[0, 0]
            else:
                col[j] = 0
        res[:, i] = col
    return res

n = torch.rand(7, 1)

o = ort(n)