import torch
import torch.nn.functional as F
from functools import partial
from functorch import vmap, vjp
from functorch import jacrev
from functorch.compile import aot_function

import numpy as np
import time
from sdf.robot_sdf import RobotSdfCollisionNet

tensor_args = {'device': 'cpu', 'dtype': torch.float32}

q_dof = 7
data = torch.load('datasets/%d_dof_data_test.pt' % q_dof).to(**tensor_args)
x = data[:, 0:q_dof + 3]
y = data[:, -q_dof:]

s = 256
n_layers = 5
skips = []
fname = '%ddof_sdf_%dx%d_mesh.pt' % (q_dof, s, n_layers)
if skips == []:
    n_layers -= 1
nn_model = RobotSdfCollisionNet(in_channels=x.shape[1], out_channels=y.shape[1], layers=[s] * n_layers, skips=skips)
nn_model.load_weights('models/' + fname, tensor_args)
nn_model.model = torch.jit.script(nn_model.model)
nn_model.model = torch.jit.optimize_for_inference(nn_model.model)
nn_model.model.to(**tensor_args)

model = nn_model.model
nelem = sum([param.nelement() for param in model.parameters()])
print(repr(model))
print("Sum of parameters:%d" % nelem)

# print(x.shape)
N_REP = 100
H = 1
N_TRAJ = 1
N_OBS = 1
tens_input = torch.tile(x[0:N_TRAJ, :], (N_OBS, 1))
for i in range(10):
    tens_input = tens_input
    # default method
    y_pred = nn_model.compute_signed_distance_wgrad(tens_input, 'closest')

t0 = time.time()
grad_map = torch.zeros(7, tens_input.shape[0], 7)

# fn = lambda x : nn_model.compute_signed_distance_wgrad2(x)
# def print_compile_fn(fx_module, args):
#      #print(fx_module)
#      return fx_module
#
# aot_fn = aot_function(fn, print_compile_fn)


nn_model.allocate_gradients(tens_input.shape[0], tensor_args)
for i in range(N_REP):
    for i in range(H):
        tens_input.requires_grad = False
        tens_input = tens_input+0.1
        y_pred = nn_model.compute_signed_distance_wgrad(tens_input, 'closest')
        #y_pred = nn_model.dist_grad_closest(tens_input)
        #y_pred = nn_model.model.forward(tens_input)
        #y_pred = nn_model.compute_signed_distance_wgrad2(tens_input)
        # y_pred = aot_fn(tens_input)
        # dists, vjp_fn = vjp(partial(nn_model.model.forward), tens_input)
        # minidxMask = torch.argmin(dists, dim=1)
        # grad_map[minidxMask, list(range(tens_input.shape[0])), minidxMask] = 1
        # ft_jacobian = (vmap(vjp_fn)(grad_map))[0].sum(0)

t1 = time.time()
print('Total time: %4.2fs' % (t1 - t0))
print('Avg freq: %4.2f Hz' % (N_REP/(t1 - t0)))

print("Time per sample: %4.10fms" % (1e6*(t1 - t0) / (N_REP * H * tens_input.shape[0])))
# print(y_pred.shape, y_test.shape)
#loss = F.l1_loss(y_pred, y, reduction='mean')
#print(torch.median(y_pred), torch.mean(y_pred))
#print(loss.item())
#dist_dif = torch.abs(y_pred - y)
#print('Mean distance difference: ', dist_dif.mean(dim=0))
#print('Mean distance std: ', dist_dif.std(dim=0))
#a = nn_model.compute_signed_distance_wgrad(x, 'mindist')
#print(a)
