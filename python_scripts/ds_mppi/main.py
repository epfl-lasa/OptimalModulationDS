import numpy as np
from fk_num import *
from fk_sym_gen import *
from plots import *
import numpy as np
from propagation import *
import torch
import time
from policy import *

# define tensor parameters (cpu or cuda:0)
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
    q_0[0] = torch.pi / 2
    q_f[0] = -torch.pi / 2
    # Robot parameters
    dh_a = torch.zeros(DOF + 1).to(**params)
    dh_a[1:] = L  # link length
    dh_params = torch.vstack((dh_a * 0, dh_a * 0, dh_a, dh_a * 0)).T
    # Obstacle spheres (x, y, z, r)
    obs = torch.tensor([[4, 0, 0, 1]]).to(**params)
    # Plotting
    r_h = init_robot_plot(dh_params, -10, 10, -10, 10)
    o_h_arr = plot_obs_init(obs)
    # Integration parameters
    A = -1 * torch.diag(torch.ones(DOF)).to(**params)
    N_traj = 10
    dt_H = 20
    dt = 0.2
    q_cur = q_0
    # jit warmup
    for i in range(30):
        all_traj = propagate_mod(q_cur, q_f, dh_params, obs, dt, 1, 1, A, dh_a)
    t0 = time.time()
    N_ITER = 0
    P = TensorPolicyMPPI(N_traj, DOF, params)
    # P.add_kernel(q_0)
    # P.add_kernel(q_0*1.1)
    # P.add_kernel(q_0*0.9)
    thr_dist = 0.4
    thr_rbf = 0.2
    while torch.norm(q_cur - q_f) > 0.1:
        # Sample random policies
        P.sample_policy()
        # Propagate modulated DS
        all_traj, closests_dist_all, kernel_val_all = propagate_mod_policy(P, q_cur, q_f, dh_params, obs, dt, dt_H,
                                                                           N_traj, A, dh_a)
        # Check trajectory for new kernel candidates
        kernel_candidates = check_traj_for_kernels(all_traj, closests_dist_all, kernel_val_all, thr_dist, thr_rbf)
        if len(kernel_candidates) > 0:
            rand_idx = torch.randint(kernel_candidates.shape[0], (1,))
            P.add_kernel(kernel_candidates[rand_idx[0]])
        # Update current robot state
        q_cur = all_traj[0, 1, :]
        cur_fk, _ = numeric_fk_model(q_cur, dh_params, 10)
        upd_r_h(cur_fk, r_h)
        # obs[0, 0] += 0.03
        # plot_obs_update(o_h_arr, obs)
        plt.pause(0.0001)
        N_ITER += 1
        if N_ITER > 1000:
            break
        # print(q_cur)
    td = time.time() - t0
    print('Time: ', td)
    print('Time per iteration: ', td / N_ITER, 'Hz: ', 1 / (td / N_ITER))
    print('Time per rollout: ', td / (N_ITER * N_traj))
    print('Time per rollout step: ', td / (N_ITER * N_traj * dt_H))
    plt.pause(1000)


if __name__ == '__main__':
    main_int()
