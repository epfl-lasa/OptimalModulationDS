import torch
import numpy as np
import sys
import time
sys.dont_write_bytecode = False
import copy

sys.path.append('../functions/')
from MPPI_toy import *
sys.path.append('../../mlp_learn/')
from sdf.robot_sdf import RobotSdfCollisionNet

# define tensor parameters (cpu or cuda:0)
params = {'device': 'cpu', 'dtype': torch.float32}



# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
def main_int():
    t00 = time.time()
    DOF = 2
    L = 3

    # Load nn model
    s = 256
    n_layers = 5
    skips = []
    fname = '%ddof_sdf_%dx%d_toy.pt' % (DOF, s, n_layers)
    if skips == []:
        n_layers -= 1
    nn_model = RobotSdfCollisionNet(in_channels=DOF+2, out_channels=1, layers=[s] * n_layers, skips=skips)
    nn_model.load_weights('../../mlp_learn/models/' + fname, params)
    nn_model.model.to(**params)
    # prepare models: standard (used for AOT implementation), jit, jit+quantization
    nn_model.model_jit = nn_model.model

    # nn_model.model_jit_q = torch.quantization.quantize_dynamic(
    #     nn_model.model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    # )

    nn_model.model_jit = torch.jit.script(nn_model.model_jit)
    nn_model.model_jit = torch.jit.optimize_for_inference(nn_model.model_jit)

    # nn_model.model_jit_q = torch.jit.script(nn_model.model_jit)
    # nn_model.model_jit_q = torch.jit.optimize_for_inference(nn_model.model_jit)

    nn_model.update_aot_lambda()

    #nn_model.model.eval()
    # Initial state
    q_0 = torch.tensor([-5, 0]).to(**params)
    q_f = torch.tensor([8, 0]).to(**params)
    # dummy dh parameters
    dh_params = torch.zeros(4, 4)
    # obstacles
    # generate arc consisting of spheres of radius 1
    t = torch.linspace(-np.pi / 3, np.pi / 3, 20)
    c = torch.tensor([0, 0])
    r_arc = 3
    r_sph = 1
    obs_arr = torch.vstack([c[0] + r_arc * torch.cos(t),
                            c[1] + r_arc * torch.sin(t),
                            r_sph * torch.ones(t.shape[0])]).transpose(0, 1)
    obs = torch.tensor(obs_arr).to(**params)
    # Plotting
    r_h = init_toy_plot(-10, 10, -10, 10)
    c_h = init_kernel_means(100)
    o_h_arr = plot_obs_init(obs)
    # Integration parameters
    A = -1 * torch.diag(torch.ones(DOF)).to(**params) #nominal DS
    N_traj = 100                 # number of trajectories in exploration sampling
    dt_H = 10                   # horizon length in exploration sampling
    dt = 0.5                    # integration timestep in exploration sampling
    dt_sim = 0.1                # integration timestep for actual robot motion

    N_ITER = 0
    # kernel adding thresholds
    dst_thr = 0.5               # distance to collision (everything below - adds a kernel)
    thr_rbf_add = 0.05          # distance to closest kernel (l2 norm of 7d vector difference)
    thr_dot_add = -0.9

    #primary MPPI to sample naviagtion policy
    mppi = MPPI(q_0, q_f, dh_params, obs, dt, dt_H, N_traj, A, 0, nn_model, 1)
    mppi.Policy.sigma_c_nominal = 0.1
    mppi.Policy.alpha_s = 0.75
    mppi.Policy.policy_upd_rate = 0.5
    mppi.dst_thr = dst_thr/2      # substracted from actual distance (added threshsold)
    mppi.ker_thr = 0.5         # used to create update mask for policy means
    mppi.ignored_links = []
    mppi.Cost.q_min = -10*torch.ones(DOF).to(**params)
    mppi.Cost.q_max =  10*torch.ones(DOF).to(**params)
    #set up second mppi to move the robot
    mppi_step = MPPI(q_0, q_f, dh_params, obs, dt_sim, 1, 1, A, 0, nn_model, 1)
    mppi_step.Policy.alpha_s *= 0
    mppi_step.ignored_links = []

    best_idx = -1
    t0 = time.time()
    print('Init time: %4.2fs' % (t0 - t00))
    PROFILING = False
    while torch.norm(mppi.q_cur - q_f) > 0.1:
        t_iter = time.time()
        # Sample random policies
        mppi.Policy.sample_policy()
        # Propagate modulated DS

        with record_function("TAG: general propagation"):
            all_traj, closests_dist_all, kernel_val_all, dotproducts_all = mppi.propagate()

        with record_function("TAG: cost calculation"):
            # Calculate cost
            cost = mppi.get_cost()
            best_idx = torch.argmin(cost)
            mppi.shift_policy_means()
        # Check trajectory for new kernel candidates and add policy kernels
        kernel_candidates = mppi.Policy.check_traj_for_kernels(all_traj, closests_dist_all, dotproducts_all,
                                                               dst_thr - mppi.dst_thr, thr_rbf_add, thr_dot_add)

        if len(kernel_candidates) > 0:
            rand_idx = torch.randint(kernel_candidates.shape[0], (1,))[0]
            closest_candidate_norm, closest_idx = torch.norm(kernel_candidates - mppi.q_cur, 2, -1).min(dim=0)
            if closest_candidate_norm < 1e-1:
                idx_to_add = closest_idx
            else:
                idx_to_add = rand_idx
            candidate = kernel_candidates[idx_to_add]
            # some mess to store the obstacle basis
            idx_i, idx_h = torch.where((all_traj == candidate).all(dim=-1))
            idx_i, idx_h = idx_i[0], idx_h[0]
            # add the kernel finally
            mppi.Policy.add_kernel(kernel_candidates[idx_to_add], closests_dist_all[idx_i, idx_h],
                                   mppi.norm_basis[idx_i, idx_h].squeeze())
            upd_toy_h(kernel_candidates[idx_to_add].to('cpu'), c_h[(mppi.Policy.n_kernels - 1) % len(c_h)])

        # Update current robot state
        mppi_step.Policy.mu_c = mppi.Policy.mu_c
        mppi_step.Policy.sigma_c = mppi.Policy.sigma_c
        if 0*best_idx:
            mppi_step.Policy.alpha_c = mppi.Policy.alpha_tmp[best_idx]
        else:
            mppi_step.Policy.alpha_c = mppi.Policy.alpha_c

        mppi_step.Policy.n_kernels = mppi.Policy.n_kernels
        mppi_step.Policy.sample_policy()
        mppi_step.q_cur = copy.copy(mppi.q_cur)
        _, _, _, _ = mppi_step.propagate()
        mppi.q_cur = mppi.q_cur + mppi_step.qdot[0, :] * dt_sim
        #print(mppi.qdot[best_idx, :] - mppi_step.qdot[0, :])
        upd_toy_h(mppi.q_cur.to('cpu'), r_h)
        r_h.set_zorder(1000)
        # obs[0, 0] += 0.03
        # plot_obs_update(o_h_arr, obs)
        plt.pause(0.0001)
        N_ITER += 1
        if PROFILING:
            torch_profiler.step()
        if N_ITER > 10000:
            break
        if N_ITER == 4:
            t0 = time.time()
        # print(q_cur)
        t_iter = time.time() - t_iter
        print(f'Iteration:{N_ITER:4d}, Time:{t_iter:4.2f}, Frequency:{1/t_iter:4.2f},',
              f' Avg. frequency:{N_ITER/(time.time()-t0):4.2f}',
              f' Kernel count:{mppi.Policy.n_kernels:4d}')
        data = {'n_kernels': mppi.Policy.n_kernels,
                'mu_c': mppi.Policy.mu_c[0:mppi.Policy.n_kernels],
                'alpha_c': mppi.Policy.alpha_c[0:mppi.Policy.n_kernels],
                'sigma_c': mppi.Policy.sigma_c[0:mppi.Policy.n_kernels],
                'norm_basis': mppi.Policy.kernel_obstacle_bases[0:mppi.Policy.n_kernels]}
        torch.save(data, 'toy_policy.pt')


    td = time.time() - t0
    print('Time: ', td)
    print('Time per iteration: ', td / N_ITER, 'Hz: ', 1 / (td / (N_ITER)))
    print('Time per rollout: ', td / (N_ITER * N_traj))
    print('Time per rollout step: ', td / (N_ITER * N_traj * dt_H))
    #print(torch_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    plt.pause(10)


if __name__ == '__main__':
    main_int()
