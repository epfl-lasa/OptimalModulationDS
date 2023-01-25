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
if 0:
    params = {'device': 'cpu', 'dtype': torch.float32}
else:
    params = {'device': 'cuda:0', 'dtype': torch.float32}


# trace_handler for pytorch profiling
def trace_handler(profiler):
    print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))


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
    # nn_model.model = torch.jit.script(nn_model.model)
    # nn_model.model = torch.jit.optimize_for_inference(nn_model.model)

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
    obs = torch.tensor([[4, 3, 0, .3],
                        [5, 0, 0, .3],
                        [4, -3, 0, .3]]).to(**params)
    # Plotting
    r_h = init_robot_plot(dh_params, -10, 10, -10, 10)
    o_h_arr = plot_obs_init(obs)
    # Integration parameters
    A = -1 * torch.diag(torch.ones(DOF)).to(**params)
    N_traj = 50
    dt_H = 30
    dt = 0.2
    q_cur = q_0
    N_ITER = 0
    # kernel adding thresholds
    thr_dist = 1
    thr_rbf = 0.1
    mppi = MPPI(q_0, q_f, dh_params, obs, dt, dt_H, N_traj, A, dh_a, nn_model)

    # jit warmup
    for i in range(50):
        a,b,c = mppi.propagate()
    t0 = time.time()
    print('Init time: %4.2fs' % (t0 - t00))
    with profile(schedule=torch.profiler.schedule(wait=10, warmup=1, active=1, repeat=1),
                 activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 on_trace_ready=trace_handler,
                 profile_memory=True, record_shapes=True, with_stack=True) as torch_profiler:
    #with open('.pytest_cache/dummy', 'w') as dummy_file: #that's an empty with statement to replace profining when unused
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

            # Update current robot state
            mppi.q_cur = all_traj[best_idx, 1, :]
            cur_fk, _ = numeric_fk_model(mppi.q_cur, dh_params, 10)
            upd_r_h(cur_fk.to('cpu'), r_h)
            # obs[0, 0] += 0.03
            # plot_obs_update(o_h_arr, obs)
            plt.pause(0.0001)
            N_ITER += 1
            torch_profiler.step()
            if N_ITER > 100:
                break
            # print(q_cur)
            t_iter = time.time() - t_iter
            print(f'Iteration:{N_ITER:4d}, Time:{t_iter:4.2f}, Frequency:{1/t_iter:4.2f},'
                  f' Avg. frequency:{N_ITER/(time.time()-t0):4.2f}')
    td = time.time() - t0
    print('Time: ', td)
    print('Time per iteration: ', td / N_ITER, 'Hz: ', 1 / (td / N_ITER))
    print('Time per rollout: ', td / (N_ITER * N_traj))
    print('Time per rollout step: ', td / (N_ITER * N_traj * dt_H))
    #print(torch_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    plt.pause(10)


if __name__ == '__main__':
    main_int()
