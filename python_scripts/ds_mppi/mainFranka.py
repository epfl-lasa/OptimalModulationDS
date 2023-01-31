from isaac_gym_helpers import *
from MPPI import *
import torch

sys.path.append('../mlp_learn/')
from sdf.robot_sdf import RobotSdfCollisionNet

# define tensor parameters (cpu or cuda:0)
if 1:
    params = {'device': 'cpu', 'dtype': torch.float32}
else:
    params = {'device': 'cuda:0', 'dtype': torch.float32}

def main_loop(gym_instance):
    ########################################
    ###        CONTROLLER SETUP          ###
    ########################################
    t00 = time.time()
    DOF = 7
    L = 1

    # Load nn model
    s = 256
    n_layers = 5
    skips = []
    #fname = '%ddof_sdf_%dx%d_mesh.pt' % (DOF, s, n_layers) # planar robot
    fname = 'franka_%dx%d.pt' % (s, n_layers)  # franka robot

    if skips == []:
        n_layers -= 1
    nn_model = RobotSdfCollisionNet(in_channels=DOF+3, out_channels=9, layers=[s] * n_layers, skips=skips)
    nn_model.load_weights('../mlp_learn/models/' + fname, params)
    nn_model.model.to(**params)
    # prepare models: standard (used for AOT implementation), jit, jit+quantization
    nn_model.model_jit = nn_model.model
    nn_model.model_jit = torch.jit.script(nn_model.model_jit)
    nn_model.model_jit = torch.jit.optimize_for_inference(nn_model.model_jit)
    nn_model.update_aot_lambda()
    #nn_model.model.eval()
    # Initial state
    q_0 = torch.tensor([-1.1, -0.08, 0, -2, -0.16,  1.6, -0.75]).to(**params)
    q_f = torch.tensor([1.1, -0.08, 0, -2, -0.16,  1.6, -0.75]).to(**params)

    # Robot parameters
    dh_a = torch.zeros(DOF + 1).to(**params)
    dh_a[1:] = L  # link length
    dh_a = torch.tensor([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0])        # "r" in matlab
    dh_d = torch.tensor([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107])       # "d" in matlab
    dh_alpha = torch.tensor([0, -pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2, 0])  # "alpha" in matlab
    dh_params = torch.vstack((dh_d, dh_a*0, dh_a, dh_alpha)).T          # (d, theta, a (or r), alpha)
    # Obstacle spheres (x, y, z, r)
    # obs = torch.tensor([[6, 2, 0, .5],
    #                     [4, -1, 0, .5],
    #                     [5, 0, 0, .5]]).to(**params)

    obs = torch.tensor([[0.4, 0, 0.00, .05],
                        [0.4, 0, 0.55, .05],
                        [0.4, 0, 0.60, .05],
                        [0.4, 0, 0.65, .05],
                        [0.4, 0, 0.70, .05]]).to(**params)

    n_dummy = 1
    dummy_obs = torch.hstack((torch.zeros(n_dummy, 3)+10, torch.zeros(n_dummy, 1)+0.1)).to(**params)
    obs = torch.vstack((obs, dummy_obs))
    # Integration parameters
    A = -1 * torch.diag(torch.ones(DOF)).to(**params)
    N_traj = 20
    dt_H = 10
    dt = 0.1
    dt_sim = 0.01
    N_ITER = 0
    # kernel adding thresholds
    thr_dist = 0.5
    thr_rbf = 0.03
    mppi = MPPI(q_0, q_f, dh_params, obs, dt, dt_H, N_traj, A, dh_a, nn_model)
    mppi.Policy.sigma_c_nominal = 0.3
    mppi.Policy.alpha_s = 0.3
    mppi.Policy.policy_upd_rate = 0.5
    mppi.dst_thr = 0.05 # 5 cm
    # jit warmup
    for i in range(20):
        _, _, _ = mppi.propagate()

    # ########################################
    # ###     GYM AND SIMULATION SETUP     ###
    # ########################################
    world_instance, robot_sim, robot_ptr, env_ptr = deploy_world_robot(gym_instance, params)
    w_T_r = copy.deepcopy(robot_sim.spawn_robot_pose)

    obs_list = []
    for i, sphere in enumerate(obs):
        tmpObsDict = deploy_sphere(sphere, gym_instance, w_T_r, 'sphere_%d'%(i), params)
        obs_list.append(tmpObsDict)

    t0 = time.time()
    print('Init time: %4.2fs' % (t0 - t00))

    while torch.norm(mppi.q_cur - q_f)+1 > 0.001:
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
            #kernel_fk, _ = numeric_fk_model(kernel_candidates[rand_idx[0]], dh_params, 10)
            #upd_r_h(kernel_fk.to('cpu'), c_h[(mppi.Policy.n_kernels - 1) % len(c_h)])

        # Update current robot state
        qdot = mppi.get_qdot('best')
        mppi.q_cur = mppi.q_cur + dt_sim * qdot
        cur_fk, _ = numeric_fk_model(mppi.q_cur, dh_params, 10)

        gym_instance.step()
        q_des = mppi.q_cur
        dq_des = qdot * 0
        robot_sim.set_robot_state(q_des, dq_des, env_ptr, robot_ptr)

        N_ITER += 1
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
    time.sleep(10)


if __name__ == '__main__':
    # instantiate empty gym:
    sim_params = load_yaml(join_path(get_gym_configs_path(), 'physx.yml'))
    gym_instance = Gym(**sim_params)
    main_loop(gym_instance)


