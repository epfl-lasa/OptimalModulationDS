import torch
import torch.nn.functional as F
import numpy as np
import time
from sdf.robot_sdf import RobotSdfCollisionNet

if 1:
    params = {'device': 'cpu', 'dtype': torch.float32}
else:
    params = {'device': 'cuda:0', 'dtype': torch.float32}

data = torch.load('data.pt').to(**params)

q_dof = 2
x = torch.Tensor(data[:, 0:q_dof + 3]).to(**params)
y = torch.Tensor(data[:, -q_dof:]).to(**params)

dof = x.shape[1]
s = 64
n_layers = 3
skips = []
fname = 'sdf_%dx%d_mesh.pt'%(s,n_layers)
if skips == []:
    n_layers-=1
nn_model = RobotSdfCollisionNet(in_channels=dof, out_channels=y.shape[1], layers=[s] * n_layers, skips=skips)
nn_model.load_weights(fname, params)

nn_model.model.to(**params)

model = nn_model.model
nelem = sum([param.nelement() for param in model.parameters()])
print(repr(model))
print("Sum of parameters:%d" % nelem)


# print(x.shape)
t0 = time.time()
N_REP = 10
for i in range(N_REP):
    y_pred = model.forward(x)
t1 = time.time()
print("Time per sample: %4.10f" % ((t1 - t0)/(N_REP*x.shape[0])))
# print(y_pred.shape, y_test.shape)
loss = F.l1_loss(y_pred, y, reduction='mean')
print(torch.median(y_pred), torch.mean(y_pred))
print(loss.item())

