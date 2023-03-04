import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
import numpy as np
sys.dont_write_bytecode = False
import copy

sys.path.append('../functions/')
from MPPI_toy import *
sys.path.append('../../mlp_learn/')
from sdf.robot_sdf import RobotSdfCollisionNet

params = {'device': 'cpu', 'dtype': torch.float32}

# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
DOF = 2
L = 3

# Load nn model
s = 256
n_layers = 5
skips = []
fname = '%ddof_sdf_%dx%d_mesh.pt' % (DOF, s, n_layers)
fname = '%ddof_sdf_%dx%d_toy.pt' % (DOF, s, n_layers)
if skips == []:
    n_layers -= 1
nn_model = RobotSdfCollisionNet(in_channels=DOF+2, out_channels=1, layers=[s] * n_layers, skips=skips)
nn_model.load_weights('../../mlp_learn/models/' + fname, params)
nn_model.model.to(**params)
nn_model.model_jit = nn_model.model
nn_model.model_jit = torch.jit.script(nn_model.model_jit)
nn_model.model_jit = torch.jit.optimize_for_inference(nn_model.model_jit)
nn_model.update_aot_lambda()

q_min = -10 * torch.ones(DOF).to(**params)
q_max = 10 * torch.ones(DOF).to(**params)

# generate meshgrid within q_min, q_max
n_mg = 100
points_grid = torch.meshgrid([torch.linspace(q_min[i], q_max[i], n_mg) for i in range(DOF)])
q_tens = torch.stack(points_grid, dim=-1).reshape(-1, DOF)

# obstacles
# generate arc consisting of spheres of radius 1
t = torch.linspace(-np.pi/3, np.pi/3, 20)
c = torch.tensor([0, 0])
r_arc = 3
r_sph = 1
obs_arr = torch.vstack([c[0]+r_arc * torch.cos(t),
                        c[1]+r_arc * torch.sin(t),
                        r_sph*torch.ones(t.shape[0])]).transpose(0, 1)

# obs_arr = np.linspace([4, 3, 0, 1], [4, -3, 0, 1], 10)
obs_tens = torch.tensor(obs_arr).to(**params)

# obs_tens = torch.tensor([[0, 2, 0, 1], [2, 1, 0, 2], [2, -1, 0, 2], [0, -2, 0, 1]]).to(**params)
# obs_tens = obs_tens[0:1]
# input tensor for nn (using repeat_interleave)
nn_input = torch.hstack((q_tens.tile(obs_tens.shape[0], 1), obs_tens.repeat_interleave(q_tens.shape[0], 0)))
distances = nn_model.model(nn_input[:, 0:DOF+2]).detach()
distances = distances.min(dim=1).values # find closest link in each case
distances -= nn_input[:, -1] # substract radii
distances = distances.reshape(obs_tens.shape[0], q_tens.shape[0]).transpose(-1,0) # reshape obstacle-like
distances = distances.min(dim=1).values # find closest obstacle in each case
distances = distances.reshape(n_mg, n_mg) # reshape grid-like

# plot contour of distances
plt.figure()
# custom listed colormap
carr = np.linspace([.1, .1, 1, 1], [1, 1, 1, 1], 256)

# DS
A = torch.tensor([[-1, 0], [0, -1]]).to(**params)
attractor = torch.tensor([8, 0]).to(**params)
# calculate ds flow
ds_flow = A @ (q_tens - attractor).transpose(0, 1)
# plot streamplot
# plt.streamplot(points_grid[0][:, 0].numpy(),
#                points_grid[1][0, :].numpy(),
#                ds_flow[0].reshape(n_mg, n_mg).numpy().T,
#                ds_flow[1].reshape(n_mg, n_mg).numpy().T,
#                density=2, color='b', linewidth=0.5, arrowstyle='->')
#
# plt.plot(attractor[0], attractor[1], 'r*', markersize=10)
#plt.show()

q_0 = torch.tensor([-5, 0]).to(**params)
mppi_step = MPPI(q_0, attractor, torch.zeros(4, 4), obs_tens, 0.1, 1, 1, A, 0, nn_model, 1)
ds_flow = torch.zeros(q_tens.shape).to(**params)
for i, point in enumerate(q_tens):
    mppi_step.q_cur = point
    _, _, _, _ = mppi_step.propagate()
    ds_flow[i] = mppi_step.qdot[0, :]
    print(i)

plt.streamplot(points_grid[0][:, 0].numpy(),
               points_grid[1][0, :].numpy(),
               ds_flow[:,0].reshape(n_mg, n_mg).numpy().T,
               ds_flow[:,1].reshape(n_mg, n_mg).numpy().T,
               density=2, color='b', linewidth=0.5, arrowstyle='->')

trajs_init = np.linspace([-10,10], [-10, -10], 9)
trajs = []

mppi_traj = MPPI(q_0, attractor, torch.zeros(4, 4), obs_tens, 0.1, 1000, 1, A, 0, nn_model, 1)
for point in trajs_init:
    mppi_traj.q_cur = torch.tensor(point).to(**params)
    traj, _, _, _ = mppi_traj.propagate()
    trajs.append(traj)

for traj in trajs:
    plt.plot(traj[0][:, 0], traj[0][:, 1], color=[0.85, 0.32, 0.1], linewidth=2)
plt.plot(attractor[0], attractor[1], 'r*', markersize=10)

#finally place obstacles
plt.contourf(points_grid[0], points_grid[1], distances, levels=1000, cmap=ListedColormap(carr), vmax=0.5)
# contour zero lvl
plt.contour(points_grid[0], points_grid[1], distances, levels=[0], colors='k')
plt.gca().set_aspect('equal', adjustable='box')

plt.show()
