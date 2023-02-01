import torch

from propagation import *
import sys
import cProfile
import pstats

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

# trace_handler for pytorch profiling
def trace_handler(profiler):
    print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    profiler.export_chrome_trace("trace_CPU.json")


torch.manual_seed(0)
torch.cuda.manual_seed(0)
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
    nn_model.load_weights('../mlp_learn/models/' + fname, params)
    nn_model.model.to(**params)
    # prepare models: standard (used for AOT implementation), jit, jit+quantization
    nn_model.model_jit = nn_model.model

    nn_model.model_jit_q = torch.quantization.quantize_dynamic(
        nn_model.model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    )

    nn_model.model_jit = torch.jit.script(nn_model.model_jit)
    nn_model.model_jit = torch.jit.optimize_for_inference(nn_model.model_jit)

    nn_model.model_jit_q = torch.jit.script(nn_model.model_jit_q)
    nn_model.model_jit_q = torch.jit.optimize_for_inference(nn_model.model_jit_q)

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
                        [4.5, -1, 0, .5],
                        [5, 0, 0, .5]]).to(**params)
    n_dummy = 1
    dummy_obs = torch.hstack((torch.zeros(n_dummy, 3)+6, torch.zeros(n_dummy, 1)+0.1)).to(**params)
    obs = torch.vstack((obs, dummy_obs))
    # Plotting
    r_h = init_robot_plot(dh_params, -10, 10, -10, 10)
    c_h = init_kernel_means(100)
    o_h_arr = plot_obs_init(obs)
    # Integration parameters
    A = -1 * torch.diag(torch.ones(DOF)).to(**params)
    N_traj = 50
    dt_H = 20
    dt = 0.2
    dt_sim = 0.05
    q_cur = q_0
    N_ITER = 0
    # kernel adding thresholds
    thr_dist = 0.5
    thr_rbf = 1e-3
    mppi = MPPI(q_0, q_f, dh_params, obs, dt, dt_H, N_traj, A, dh_a, nn_model)
    mppi.Policy.sigma_c_nominal = 0.3
    mppi.Policy.alpha_s = 0.3
    mppi.Policy.policy_upd_rate = 0.5
    mppi.dst_thr = 0.4
    # jit warmup
    for i in range(20):
        _, _, _ = mppi.propagate()
    #     _ = mppi.nn_model.model_jit.forward(torch.randn(N_traj*obs.shape[0], 10).to(**params))
    #     _ = mppi.nn_model.model_jit_q.forward(torch.randn(N_traj*obs.shape[0], 10).to(**params))
    #     _, _, _ = mppi.nn_model.dist_grad_closest(torch.randn(N_traj, 10).to(**params))
    #     _, _, _ = mppi.nn_model.dist_grad_closest_aot(torch.randn(N_traj, 10).to(**params))

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
                all_traj, closests_dist_all, kernel_val_all = mppi.propagate()

            with record_function("TAG: cost calculation"):
                # Calculate cost
                cost = mppi.get_cost()
                best_idx = torch.argmin(cost)
                mppi.shift_policy_means()
            # Check trajectory for new kernel candidates and add policy kernels
            kernel_candidates = check_traj_for_kernels(all_traj, closests_dist_all, kernel_val_all, thr_dist, thr_rbf)
            if len(kernel_candidates) > 0:
                rand_idx = torch.randint(kernel_candidates.shape[0], (1,))
                mppi.Policy.add_kernel(kernel_candidates[rand_idx[0]])
                kernel_fk, _ = numeric_fk_model(kernel_candidates[rand_idx[0]], dh_params, 10)
                upd_r_h(kernel_fk.to('cpu'), c_h[(mppi.Policy.n_kernels - 1) % len(c_h)])

            # Update current robot state
            #mppi.q_cur = all_traj[best_idx, 1, :]
            mppi.q_cur = mppi.q_cur + mppi.qdot[best_idx, :] * dt_sim
            cur_fk, _ = numeric_fk_model(mppi.q_cur, dh_params, 10)
            upd_r_h(cur_fk.to('cpu'), r_h)
            r_h.set_zorder(10000)
            # obs[0, 0] += 0.03
            # plot_obs_update(o_h_arr, obs)
            plt.pause(0.0001)
            N_ITER += 1
            if PROFILING:
                torch_profiler.step()
            if N_ITER > 10000:
                break
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
