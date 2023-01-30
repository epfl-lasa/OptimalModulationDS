import copy, time, argparse, os, sys, math, numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array


from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import *
from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform

import torch
from MPPI import *
sys.path.append('../mlp_learn/')
from sdf.robot_sdf import RobotSdfCollisionNet

# define tensor parameters (cpu or cuda:0)
if 1:
    params = {'device': 'cpu', 'dtype': torch.float32}
else:
    params = {'device': 'cuda:0', 'dtype': torch.float32}

def main_loop(gym_instance):
    ########################################
    ###     GYM AND SIMULATION SETUP     ###
    ########################################
    ## create robot simulation
    robot_yml = 'content/configs/gym/franka.yml'
    sim_params = load_yaml(robot_yml)['sim_params']
    sim_params['asset_root'] = os.path.dirname(__file__)+'/content/assets'
    sim_params['collision_model'] = None
    # task_file = 'franka_reacher_env2.yml'
    gym = gym_instance.gym
    sim = gym_instance.sim
    robot_sim = RobotSim(gym_instance=gym_instance.gym, sim_instance=gym_instance.sim, **sim_params, device=params['device'])
    robot_pose = sim_params['robot_pose']
    env_ptr = gym_instance.env_list[0]
    robot_ptr = robot_sim.spawn_robot(env_ptr, robot_pose, coll_id=2)

    # create world
    world_yml = 'content/configs/gym/collision_demo.yml'
    world_params = load_yaml(world_yml)
    # get pose
    w_T_r = copy.deepcopy(robot_sim.spawn_robot_pose)

    w_T_robot = torch.eye(4)
    quat = torch.tensor([w_T_r.r.w, w_T_r.r.x, w_T_r.r.y, w_T_r.r.z]).unsqueeze(0)
    rot = quaternion_to_matrix(quat)
    w_T_robot[0, 3] = w_T_r.p.x
    w_T_robot[1, 3] = w_T_r.p.y
    w_T_robot[2, 3] = w_T_r.p.z
    w_T_robot[:3, :3] = rot[0]
    world_instance = World(gym, sim, env_ptr, world_params, w_T_r=w_T_r)


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
    fname = '%ddof_sdf_%dx%d_mesh.pt' % (DOF, s, n_layers)
    if skips == []:
        n_layers -= 1
    nn_model = RobotSdfCollisionNet(in_channels=DOF+3, out_channels=DOF, layers=[s] * n_layers, skips=skips)
    nn_model.load_weights('../mlp_learn/models/' + fname, params)
    nn_model.model.to(**params)
    # prepare models: standard (used for AOT implementation), jit, jit+quantization
    nn_model.model_jit = nn_model.model
    nn_model.model_jit = torch.jit.script(nn_model.model_jit)
    nn_model.model_jit = torch.jit.optimize_for_inference(nn_model.model_jit)
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
                        [4, -1, 0, .5],
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
    q_cur = q_0
    N_ITER = 0
    # kernel adding thresholds
    thr_dist = 0.5
    thr_rbf = 0.03
    mppi = MPPI(q_0, q_f, dh_params, obs, dt, dt_H, N_traj, A, dh_a, nn_model)
    mppi.Policy.sigma_c_nominal = 0.3
    mppi.Policy.alpha_s = 0.3
    mppi.Policy.policy_upd_rate = 0.5
    mppi.dst_thr = 0.4
    # jit warmup
    for i in range(20):
        _, _, _ = mppi.propagate()

    t0 = time.time()
    print('Init time: %4.2fs' % (t0 - t00))
    PROFILING = False

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
        mppi.q_cur = all_traj[best_idx, 1, :]
        cur_fk, _ = numeric_fk_model(mppi.q_cur, dh_params, 10)

        gym_instance.step()
        q_des = mppi.q_cur
        dq_des = q_des * 0
        robot_sim.set_robot_state(q_des, dq_des, env_ptr, robot_ptr)

        upd_r_h(cur_fk.to('cpu'), r_h)
        r_h.set_zorder(10000)
        # obs[0, 0] += 0.03
        # plot_obs_update(o_h_arr, obs)
        plt.pause(0.0001)
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
    plt.pause(10)


if __name__ == '__main__':
    # instantiate empty gym:
    sim_params = load_yaml(join_path(get_gym_configs_path(), 'physx.yml'))
    gym_instance = Gym(**sim_params)
    main_loop(gym_instance)


