from propagation import *
import sys

#
from torch.profiler import record_function

from policy import *
from propagation import *
from MPPI import *
sys.path.append('../mlp_learn/')
from sdf.robot_sdf import RobotSdfCollisionNet

# define tensor parameters (cpu or cuda:0)
if 1:
    params = {'device': 'cpu', 'dtype': torch.float32}
else:
    params = {'device': 'cuda:0', 'dtype': torch.float32}


def main_int():
    DOF = 2
    L = 3

    # Load nn model
    s = 256
    n_layers = 5
    skips = []
    fname = '%ddof_sdf_%dx%d_mesh.pt' % (DOF, s, n_layers)
    if skips == []:
        n_layers -= 1
    nn_model = RobotSdfCollisionNet(in_channels=DOF+3, out_channels=DOF, layers=[s] * n_layers, skips=skips)
    nn_model.load_weights('../mlp_learn/models/' + fname, params)
    nn_model.model.to(**params)

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
    obs = torch.tensor([[6, 0, 0, 1]]).to(**params)
    # Plotting
    r_h = init_robot_plot(dh_params, -10, 10, -10, 10)
    o_h_arr = plot_obs_init(obs)
    # Integration parameters
    A = -1 * torch.diag(torch.ones(DOF)).to(**params)
    N_traj = 10
    dt_H = 30
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
    thr_dist = 1
    thr_rbf = 0.1
    #all_traj, closests_dist_all, kernel_val_all = propagate_mod_policy_nn(P, q_cur, q_f, dh_params, obs, dt, dt_H,
    #                                                                   N_traj, A, dh_a, nn_model)
    mppi = MPPI(P, q_0, q_f, dh_params, obs, dt, dt_H, N_traj, A, dh_a, nn_model)
    while torch.norm(q_cur - q_f) > 0.1:
        t_iter = time.time()
        # Sample random policies
        mppi.P.sample_policy()
        # Propagate modulated DS

        #all_traj, closests_dist_all, kernel_val_all = propagate_mod_policy(P, q_cur, q_f, dh_params, obs, dt, dt_H,
        #                                                                   N_traj, A, dh_a)
        #all_traj, closests_dist_all, kernel_val_all = propagate_mod_policy_nn(P, q_cur, q_f, dh_params, obs, dt, dt_H,
        #                                                               N_traj, A, dh_a, nn_model)
        all_traj, closests_dist_all, kernel_val_all = mppi.propagate()

        # Check trajectory for new kernel candidates
        with record_function("kernels candidate check"):
            kernel_candidates = check_traj_for_kernels(all_traj, closests_dist_all, kernel_val_all, thr_dist, thr_rbf)
        if len(kernel_candidates) > 0:
            rand_idx = torch.randint(kernel_candidates.shape[0], (1,))
            mppi.P.add_kernel(kernel_candidates[rand_idx[0]])
        # Update current robot state
        mppi.q_cur = all_traj[0, 1, :]
        cur_fk, _ = numeric_fk_model(mppi.q_cur, dh_params, 10)
        upd_r_h(cur_fk.to('cpu'), r_h)

        # obs[0, 0] += 0.03
        # plot_obs_update(o_h_arr, obs)
        plt.pause(0.0001)
        N_ITER += 1
        if N_ITER > 100:
            break
        # print(q_cur)
        t_iter = time.time() - t_iter
        print(f'Iteration:{N_ITER:4d}, Time:{t_iter:4.2f}, Frequency:{1/t_iter:4.2f}')
    td = time.time() - t0
    print('Time: ', td)
    print('Time per iteration: ', td / N_ITER, 'Hz: ', 1 / (td / N_ITER))
    print('Time per rollout: ', td / (N_ITER * N_traj))
    print('Time per rollout step: ', td / (N_ITER * N_traj * dt_H))
    plt.pause(1000)


if __name__ == '__main__':
    # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    #     main_int()
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    main_int()