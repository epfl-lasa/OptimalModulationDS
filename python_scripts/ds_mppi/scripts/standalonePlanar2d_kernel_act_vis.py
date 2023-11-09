import matplotlib.pyplot as plt
import torch

import sys
sys.dont_write_bytecode = False
import copy

sys.path.append('../functions/')
from MPPI import *
sys.path.append('../../mlp_learn/')
from sdf.robot_sdf import RobotSdfCollisionNet
from LinDS import *
from matplotlib.colors import ListedColormap
import matplotlib as mpl
# define tensor parameters (cpu or cuda:0)
if 1:
    params = {'device': 'cpu', 'dtype': torch.float32}
else:
    params = {'device': 'cuda:0', 'dtype': torch.float32}

# trace_handler for pytorch profiling
def trace_handler(profiler):
    print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    profiler.export_chrome_trace("trace_CPU.json")

def furthest_points(data, num_points):
    assert num_points <= data.shape[0]

    # Initialize chosen_data as mean of all points
    chosen_data = data.mean(dim=0, keepdim=True)

    for _ in range(num_points):
        dist = ((data[:, None] - chosen_data)**2).sum(-1).min(-1)[0]
        chosen_index = dist.argmax().item()
        chosen_data = torch.cat([chosen_data, data[chosen_index, None]])

    return chosen_data


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

    nn_model2 = RobotSdfCollisionNet(in_channels=DOF+3, out_channels=DOF, layers=[s] * n_layers, skips=skips)
    nn_model2.load_weights('../../mlp_learn/models/' + fname, params)
    nn_model2.model.to(**params)
    nn_model2.model_jit = nn_model2.model
    nn_model2.model_jit = torch.jit.script(nn_model2.model_jit)
    nn_model2.model_jit = torch.jit.optimize_for_inference(nn_model2.model_jit)
    nn_model2.update_aot_lambda()

    #nn_model.model.eval()
    # Initial state
    q_0 = torch.zeros(DOF).to(**params)
    q_f = torch.zeros(DOF).to(**params)
    q_0[0] = torch.pi / 2
    q_f[0] = -torch.pi / 2
    q_0 = torch.tensor([-3.14,  0]).to(**params)
    q_f = torch.tensor([3.14, 0]).to(**params)

    # Robot parameters
    dh_a = torch.zeros(DOF + 1).to(**params)
    dh_a[1:] = L  # link length
    dh_params = torch.vstack((dh_a * 0, dh_a * 0, dh_a, dh_a * 0)).T
    # Obstacle spheres (x, y, z, r)
    # obs = torch.tensor([[5, 1, 0, .5],
    #                     [16, 0, 0, .5],
    #                     [15, -1, 0, .5]]).to(**params)
    # obs = torch.tensor([[6.0, 0.5, 0, .5],[6.0, 0.0, 0, .5],[6.0, -0.5, 0, .5],
    #                     [1.5, 1.25, 0, .5]]).to(**params)
    obs = torch.tensor([[6.0, 0.0, 0, .5],
                        [20.0, 4.5, 0, .5]]).to(**params)

    n_dummy = 0
    if n_dummy > 0:
        dummy_obs = torch.hstack((torch.zeros(n_dummy, 3)+6, torch.zeros(n_dummy, 1)+0.1)).to(**params)
        obs = torch.vstack((obs, dummy_obs))
    # Plotting
    r_h = init_robot_plot(dh_params, -10, 10, -10, 10)
    c_h = init_kernel_means(100)
    o_h_arr = plot_obs_init(obs)

    goal_h, = plt.plot([], [], 'o-', color=[0.8500, 0.3250, 0.0980], linewidth=1, markersize=5)
    goal_fk, _ = numeric_fk_model(q_f, dh_params, 2)
    #upd_r_h(goal_fk, goal_h)

    jpos_h = init_jpos_plot(-1.1 * np.pi, 1.1 * np.pi, -1.1 * np.pi, 1.1 * np.pi)
    #plt.plot(q_f[0], q_f[1], '*', color=[0.8500, 0.3250, 0.0980], markersize=7, zorder=1000)
    ## plot obstacles in joint space [only for 2d case]
    N_MESHGRID = 100
    points_grid = torch.meshgrid([torch.linspace(-np.pi, np.pi, N_MESHGRID) for i in range(DOF)])
    q_tens = torch.stack(points_grid, dim=-1).reshape(-1, DOF).to(**params)
    nn_input = torch.hstack((q_tens.tile(obs.shape[0], 1), obs.repeat_interleave(q_tens.shape[0], 0)))
    nn_dist = nn_model.model_jit(nn_input[:, 0:-1])
    nn_dist -= nn_input[:, -1].unsqueeze(1)
    mindist, _ = nn_dist.min(1)
    mindist_obst = mindist.reshape(-1, N_MESHGRID, N_MESHGRID).detach().cpu().numpy()
    mindist_all = mindist_obst.min(0)
    carr = np.linspace([.1, .1, 1, 1], [1, 1, 1, 1], 256)
    fig = plt.figure(2)
    zero_contour = plt.contour(points_grid[0], points_grid[1], mindist_all, levels=[0], colors='r')
    ker_vis1 = []
    ker_vis2 = []
    ker_contours = []
    ker_val_arr = []
    ds_h = None
    # Integration parameters
    A = -1 * torch.diag(torch.ones(DOF)).to(**params) #nominal DS
    DS1 = LinDS(q_f)
    DS2 = LinDS(q_0)
    DS1.lin_thr = 0.05
    DS2.lin_thr = 0.05
    DS_ARRAY = [DS1, DS2]

    N_traj = 10                 # number of trajectories in exploration sampling
    dt_H = 60                   # horizon length in exploration sampling
    dt = 0.05                    # integration timestep in exploration sampling
    dt_sim = 0.05                # integration timestep for actual robot motion
    plt.figure(2)
    traj_h = []
    for i in range(N_traj):
        traj_h.append(plt.plot([], [], '-', color=[0.3010, 0.7450, 0.9330], linewidth=1.5))
    N_ITER = 0
    # kernel adding thresholds
    dst_thr = 0.5               # distance to collision (everything below - adds a kernel)
    thr_rbf_add = 0.05          # distance to closest kernel (l2 norm of 7d vector difference)
    thr_dot_add = -0.9

    #primary MPPI to sample naviagtion policy
    n_closest_obs = 1
    mppi = MPPI(q_0, q_f, dh_params, obs, dt, dt_H, N_traj, DS_ARRAY, dh_a, nn_model, n_closest_obs)
    mppi.Policy.sigma_c_nominal = 0.5
    mppi.Policy.alpha_s = 2
    mppi.Policy.policy_upd_rate = 0.5
    mppi.dst_thr = dst_thr/2      # substracted from actual distance (added threshsold)
    mppi.ker_thr = 1e-1         # used to create update mask for policy means
    mppi.ignored_links = []
    mppi.Cost.q_min = -0.99*3.14*torch.ones(DOF).to(**params)
    mppi.Cost.q_max =  0.99*3.14*torch.ones(DOF).to(**params)
    #set up second mppi to move the robot
    mppi_step = MPPI(q_0, q_f, dh_params, obs, dt_sim, 1, 1, DS_ARRAY, dh_a, nn_model, n_closest_obs)

    mppi_step.Policy.alpha_s *= 0
    mppi_step.ignored_links = []

    mppi_vis = MPPI(q_0, q_f, dh_params, obs, dt_sim, 1, N_MESHGRID*N_MESHGRID, DS_ARRAY, dh_a, nn_model2, n_closest_obs)
    policy = torch.load('toy_policy2.pt')
    mppi_vis.Policy.update_with_data(policy)
    mppi_vis.Policy.alpha_s *= 0
    mppi_vis.Policy.sample_policy()  # samples a new policy using planned means and sigmas
    mppi_vis.ignored_links = []

    mppi_vis.q_cur = q_tens

    trajs_vis, closest_dist_vis, kernel_val_all_vis, dotproducts_all_vis, kernel_acts_vis = mppi_vis.propagate()
    for i in range(mppi_vis.Policy.n_kernels):
        if len(ker_contours) <= i:
            kernel_values_raw = (kernel_val_all_vis[:, :, i])# * kernel_acts_vis)
            kernel_values = kernel_values_raw.reshape(N_MESHGRID, N_MESHGRID)
            # kernel_values[closest_dist_vis[:,:,i] < 0] = 0
            kernel_values[kernel_values > 0.95] = 1
            kernel_values[kernel_values < 0.1] = 0

            nrange = 500
            carr = np.linspace(np.array([1, 1, 1, 0]), np.array([.46, .67, .18, 1]), nrange)
            ker_contours.append(plt.contourf(points_grid[0], points_grid[1], kernel_values,
                         levels=nrange,
                         cmap=ListedColormap(carr), vmin=0, vmax=1))

    # first clean up all existing kernels from plots
    for i_k in range(len(ker_vis1)):
        for k_vis in ker_vis1[i_k]:
            k_vis.remove()
        ker_vis2[i_k].remove()
    ker_vis1 = []
    ker_vis2 = []
    # now plot latest policy
    for i_k in range(mppi_vis.Policy.n_kernels):
        # visualize policy
        k_c = mppi_vis.Policy.mu_c[i_k]
        k_dir = mppi_vis.Policy.alpha_c[i_k]/torch.norm(mppi_vis.Policy.alpha_c[i_k])*0.4
        ker_vis1.append(plt.plot(k_c[0], k_c[1], 'gh', markersize=5, zorder=1000))
        ker_vis2.append(plt.arrow(k_c[0], k_c[1], k_dir[0], k_dir[1], color='g', width=0.05, zorder=999))
        # ker_vis1[i_k] = plt.plot(k_c[0], k_c[1], 'gh', markersize=5)
        # ker_vis2[i_k] = plt.arrow(k_c[0], k_c[1], k_dir[0], k_dir[1], color='g', width=0.1)
        kernel_fk, _ = numeric_fk_model(k_c, dh_params, 2)
        upd_r_h(kernel_fk.to('cpu'), c_h[(mppi_vis.Policy.n_kernels - 1) % len(c_h)])
    clrs = furthest_points(torch.rand(1000,3), 100)
    q_within = q_tens[(kernel_values_raw > 0.3).squeeze(), :]
    q_plot = furthest_points(q_within, 4)
    for q_i, q_c in enumerate(q_plot[1:]):
        kernel_fk, _ = numeric_fk_model(q_c, dh_params, 2)
        idx_k = (mppi_vis.Policy.n_kernels + q_i) % len(c_h)
        upd_r_h(kernel_fk.to('cpu'), c_h[idx_k])
        clr = clrs[q_i+1].numpy()
        c_h[idx_k].set_color(clr)
        plt.figure(2)
        plt.plot(q_c[0], q_c[1], '.', color=clr, markersize=7, zorder=1000)

    #print(torch_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    plt.pause(30)


if __name__ == '__main__':
    main_int()

