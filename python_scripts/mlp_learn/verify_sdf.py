import torch
import torch.nn.functional as F
import numpy as np
import time
from sdf.robot_sdf import RobotSdfCollisionNet

tensor_args = {'device': 'cpu:0', 'dtype': torch.float32}

q_dof = 2
data = torch.load('datasets/%d_dof_data_test.pt' % q_dof).to(**tensor_args)
x = data[:, 0:q_dof + 3]
y = data[:, -q_dof:]

s = 128
n_layers = 3
skips = []
fname = '%ddof_sdf_%dx%d_mesh.pt' % (q_dof, s, n_layers)
if skips == []:
    n_layers -= 1
nn_model = RobotSdfCollisionNet(in_channels=x.shape[1], out_channels=y.shape[1], layers=[s] * n_layers, skips=skips)
nn_model.load_weights('models/' + fname, tensor_args)

nn_model.model.to(**tensor_args)

model = nn_model.model
nelem = sum([param.nelement() for param in model.parameters()])
print(repr(model))
print("Sum of parameters:%d" % nelem)

# print(x.shape)
t0 = time.time()
N_REP = 1
for i in range(N_REP):
    y_pred = model.forward(x)
t1 = time.time()
print(t1 - t0)
print("Time per sample: %4.10f" % ((t1 - t0) / (N_REP * x.shape[0])))
# print(y_pred.shape, y_test.shape)
loss = F.l1_loss(y_pred, y, reduction='mean')
print(torch.median(y_pred), torch.mean(y_pred))
print(loss.item())
dist_dif = torch.abs(y_pred - y)
print('Mean distance difference: ', dist_dif.mean(dim=0))
print('Mean distance std: ', dist_dif.std(dim=0))
a = nn_model.compute_signed_distance_wgrad(x, 'mindist')
print(a)
