import numpy as np
from scipy.io import loadmat
import torch
import copy
import time
seds = loadmat('../content/seds_left.mat')
Mu = torch.tensor(seds['Mu'])
Sigma = torch.tensor(seds['Sigma'])
Priors = torch.tensor(seds['Priors'])
xT = torch.tensor(seds['xT'])


def gaussPDF(Data, Mu, Sigma):
    nbVar, nbData = Data.shape
    Data = Data.t() - Mu
    prob = torch.sum((Data @ torch.inverse(Sigma)) * Data, dim=1)
    prob = torch.exp(-0.5 * prob) / torch.sqrt((2 * torch.tensor(torch.pi)**nbVar) * (torch.abs(torch.det(Sigma)) + torch.tensor(1e-100)))
    return prob.squeeze()


def GMR(Priors, Mu, Sigma, x, in_dims, out_dims):
    nbData = x.shape[1]
    nbVar = Mu.shape[0]
    nbStates = Sigma.shape[2]
    dim = len(in_dims)
    Pxi = torch.zeros(nbData, nbStates)
    beta = torch.zeros(nbData, nbStates)
    y_tmp = torch.zeros(len(out_dims), nbData, nbStates)

    for i in range(nbStates):
        Pxi[:, i] = Priors[i] * gaussPDF(x[in_dims, :], Mu[in_dims, i], Sigma[in_dims, :][:, in_dims, i])

    beta = Pxi / torch.sum(Pxi, dim=1, keepdim=True)
    beta = beta.nan_to_num()
    beta = torch.clamp(beta, min=1e-8)
    for j in range(nbStates):
        y_tmp[:, :, j] = torch.tile(Mu[dim:, j].unsqueeze(1), (1, nbData)) + \
                         Sigma[dim:, :dim, j] @ torch.inverse(Sigma[:dim, :dim, j]) @ (x - torch.tile(Mu[:dim, j].unsqueeze(1), (1, nbData)))

    beta_tmp = beta.reshape((1, *beta.shape))
    y_tmp2 = torch.tile(beta_tmp, (dim, 1, 1)) * y_tmp
    y = y_tmp2.sum(dim=2)

    return y


# Test case
dof = int(Mu.shape[0]/2)
nbData = 1
Data = torch.randn(dof, nbData)*0
Data = torch.tensor([0, 0, 0, 0, 0, 0, 0]).unsqueeze(1)+0.1
prob = gaussPDF(Data, Mu[:dof, 0], Sigma[:dof, :dof, 0])

in_dims = torch.arange(0, dof)
out_dims = torch.arange(dof, 2*dof)
y = GMR(Priors, Mu, Sigma, Data, in_dims, out_dims)
print(y)
# some integration to figure out attractors
n_steps = 1000
dt = 1
x = copy.copy(Data)
t0 = time.time()
far_min_dx = 0.1
dst_thr = 0.1
for i in range(1000):
    dx = GMR(Priors, Mu, Sigma, x, in_dims, out_dims)
    dx_norm = torch.norm(dx, dim=0)
    if (dx_norm < far_min_dx) and (x.norm() > dst_thr):
        dx = dx / dx_norm * far_min_dx
    x = x + dx * dt
print(x)
