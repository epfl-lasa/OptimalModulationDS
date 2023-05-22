import torch

import sys
sys.dont_write_bytecode = False
import copy

sys.path.append('../functions/')
from MPPI import *
sys.path.append('../../mlp_learn/')
from sdf.robot_sdf import RobotSdfCollisionNet
from LinDS import *

# define tensor parameters (cpu or cuda:0)
if 1:
    params = {'device': 'cpu', 'dtype': torch.float32}
else:
    params = {'device': 'cuda:0', 'dtype': torch.float32}

# trace_handler for pytorch profiling
def trace_handler(profiler):
    print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    profiler.export_chrome_trace("trace_CPU.json")


# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
def main_int():
    t00 = time.time()
    DOF = 7
    L = 1

    # Load nn model
    s = 256
    n_layers = 5
    skips = []
    fname = '%ddof_sdf_%dx%d_mesh.pt' % (DOF, s, n_layers)
    if skips == []:
        n_layers -= 1
    nn_model = RobotSdfCollisionNet(in_channels=DOF+3, out_channels=DOF, layers=[s] * n_layers, skips=skips)
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
    q_0 = torch.zeros(DOF).to(**params)
    q_f = torch.zeros(DOF).to(**params)
    q_0[0] = torch.pi / 2
    q_f[0] = -torch.pi / 2
    # Robot parameters
    dh_a = torch.zeros(DOF + 1).to(**params)
    dh_a[1:] = L  # link length
    dh_params = torch.vstack((dh_a * 0, dh_a * 0, dh_a, dh_a * 0)).T
    # Obstacle spheres (x, y, z, r)
    obs = torch.tensor([[6, 2, 0, .5],
                        [2., -1, 0, .5],
                        [5, 0, 0, .5]]).to(**params)
    n_dummy = 1
    dummy_obs = torch.hstack((torch.zeros(n_dummy, 3)+6, torch.zeros(n_dummy, 1)+0.1)).to(**params)
    obs = torch.vstack((obs, dummy_obs))
    # Plotting
    r_h = init_robot_plot(dh_params, -10, 10, -10, 10)
    c_h = init_kernel_means(100)
    o_h_arr = plot_obs_init(obs)
    # Integration parameters
    A = -1 * torch.diag(torch.ones(DOF)).to(**params) #nominal DS
    DS1 = LinDS(q_f)
    DS2 = LinDS(q_0)
    DS_ARRAY = [DS1, DS2]

    N_traj = 100                 # number of trajectories in exploration sampling
    dt_H = 10                   # horizon length in exploration sampling
    dt = 0.3                    # integration timestep in exploration sampling
    dt_sim = 0.1                # integration timestep for actual robot motion

    N_ITER = 0
    # kernel adding thresholds
    dst_thr = 0.5               # distance to collision (everything below - adds a kernel)
    thr_rbf_add = 0.05          # distance to closest kernel (l2 norm of 7d vector difference)
    thr_dot_add = -0.9

    #primary MPPI to sample naviagtion policy
    mppi = MPPI(q_0, q_f, dh_params, obs, dt, dt_H, N_traj, DS_ARRAY, dh_a, nn_model, 1)
    mppi.Policy.sigma_c_nominal = 0.5
    mppi.Policy.alpha_s = 0.75
    mppi.Policy.policy_upd_rate = 0.5
    mppi.dst_thr = dst_thr/2      # substracted from actual distance (added threshsold)
    mppi.ker_thr = 1e-3         # used to create update mask for policy means
    mppi.ignored_links = []

    #set up second mppi to move the robot
    mppi_step = MPPI(q_0, q_f, dh_params, obs, dt_sim, 1, 1, DS_ARRAY, dh_a, nn_model, 1)
    mppi_step.Policy.alpha_s *= 0
    mppi_step.ignored_links = []

    best_idx = -1
    t0 = time.time()
    print('Init time: %4.2fs' % (t0 - t00))
    PROFILING = False
    with profile(schedule=torch.profiler.schedule(wait=20 if PROFILING else 1e10, warmup=1, active=3, repeat=1),
                 activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 on_trace_ready=trace_handler,
                 profile_memory=True, record_shapes=True, with_stack=True) as torch_profiler:
    # with open('.pytest_cache/dummy', 'w') as dummy_file: #that's an empty with statement to replace profining when unused
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
                kernel_fk, _ = numeric_fk_model(kernel_candidates[idx_to_add], dh_params, 2)
                upd_r_h(kernel_fk.to('cpu'), c_h[(mppi.Policy.n_kernels - 1) % len(c_h)])

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
            cur_fk, _ = numeric_fk_model(mppi.q_cur, dh_params, 10)
            upd_r_h(cur_fk.to('cpu'), r_h)
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
    td = time.time() - t0
    print('Time: ', td)
    print('Time per iteration: ', td / N_ITER, 'Hz: ', 1 / (td / (N_ITER)))
    print('Time per rollout: ', td / (N_ITER * N_traj))
    print('Time per rollout step: ', td / (N_ITER * N_traj * dt_H))
    #print(torch_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    plt.pause(10)


if __name__ == '__main__':
    main_int()
