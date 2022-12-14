import numpy as np
from fk_num import *
from fk_sym_gen import *
from plots import *
import numpy as np
from propagation import *
import torch
import time

#define tensor parameters (cpu or cuda:0)
if 1:
    params = {'device': 'cpu', 'dtype': torch.float32}
else:
    params = {'device': 'cuda:0', 'dtype': torch.float32}

def main_int():
    DOF = 7
    L = 1
    # Initial state
    q_0 = torch.zeros(DOF).to(**params)
    q_f = torch.zeros(DOF).to(**params)
    q_0[0] = torch.pi/2
    q_f[0] = -torch.pi/2
    # Robot parameters
    dh_a = torch.zeros(DOF+1).to(**params)
    dh_a[1:] = L  # link length
    dh_params = torch.vstack((dh_a*0, dh_a*0, dh_a, dh_a*0)).T
    # Obstacle spheres (x, y, z, r)
    obs = torch.tensor([[5, 0, 0, 1]]).to(**params)
    # Plotting
    r_h = init_robot_plot(dh_params, -10, 10, -10, 10)
    o_h_arr = plot_obs_init(obs)
    # Integration parameters
    A = -1*torch.diag(torch.ones(DOF)).to(**params)
    N_traj = 10
    dt_H = 20
    dt = 0.2
    q_cur = q_0
    # jit warmup
    for i in range(30):
        all_traj = propagate_mod(q_cur, q_f, dh_params, obs, dt, 1, 1, A, dh_a)
    t0 = time.time()
    for ITER in range(50):
        # Propagate modulated DS
        all_traj = propagate_mod(q_cur, q_f, dh_params, obs, dt, dt_H, N_traj, A, dh_a)
        # Update current robot state
        q_cur = all_traj[0, 1, :]
        cur_fk, _ = numeric_fk_model(q_cur, dh_params, 10)
        upd_r_h(cur_fk, r_h)
        # obs[0, 0] += 0.03
        # plot_obs_update(o_h_arr, obs)
        plt.pause(0.0001)
        #print(q_cur)
    td = time.time() - t0
    print('Time: ', td)
    print('Time per iteration: ', td/ITER, 'Hz: ', 1/(td/ITER))
    print('Time per rollout: ', td/(ITER*N_traj))
    print('Time per rollout step: ', td/(ITER*N_traj*dt_H))
    print('f')


if __name__ == '__main__':
    main_int()



