import sys, yaml
import zmq
sys.path.append('/functions/')
from isaac_gym_helpers import *
from MPPI import *
import torch
from zmq_utils import *


sys.path.append('../mlp_learn/')
from sdf.robot_sdf import RobotSdfCollisionNet

# define tensor parameters (cpu or cuda:0 or mps)
params = {'device': 'cpu', 'dtype': torch.float32}

def zmq_try_recv(val, socket):
    try:
        val = socket.recv_pyobj(flags=zmq.DONTWAIT)
    except:
        pass
    return val

def main_loop(gym_instance):

    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ########################################
    ###            ZMQ SETUP             ###
    ########################################
    context = zmq.Context()
    # socket to receive data from slow loop
    socket_receive_policy = init_subscriber(context, 'localhost', config["zmq"]["policy_port"])

    # socket to publish data to slow loop
    socket_send_state = init_publisher(context, '*', config["zmq"]["state_port"])

    # socket to receive obstacles
    socket_receive_obs = init_subscriber(context, 'localhost', config["zmq"]["obstacle_port"])

    # initialize variables
    policy_data = None
    obs = zmq_init_recv(socket_receive_obs)

    ########################################
    ###        CONTROLLER SETUP          ###
    ########################################
    t00 = time.time()
    DOF = 7

    # Load nn model
    fname = config["collision_model"]["fname"]
    nn_model = RobotSdfCollisionNet(in_channels=DOF+3, out_channels=9, layers=[256] * 4, skips=[])
    nn_model.load_weights('../mlp_learn/models/' + fname, params)
    nn_model.model.to(**params)

    # prepare models: standard (used for AOT implementation), jit, jit+quantization
    nn_model.model_jit = nn_model.model
    nn_model.model_jit = torch.jit.script(nn_model.model_jit)
    nn_model.model_jit = torch.jit.optimize_for_inference(nn_model.model_jit)
    nn_model.update_aot_lambda()
    #nn_model.model.eval()
    # Initial state
    q_0 = torch.tensor(config['general']['q_0']).to(**params)
    q_f = torch.tensor(config['general']['q_f']).to(**params)

    # Robot parameters
    dh_a = torch.tensor([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0])        # "r" in matlab
    dh_d = torch.tensor([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107])       # "d" in matlab
    dh_alpha = torch.tensor([0, -pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2, 0])  # "alpha" in matlab
    dh_params = torch.vstack((dh_d, dh_a*0, dh_a, dh_alpha)).T.to(**params)          # (d, theta, a (or r), alpha)
    # Obstacle spheres (x, y, z, r)
    # T-bar
    t1 = torch.tensor([0.4, -0.3, 0.75, .05])
    t2 = t1 + torch.tensor([0, 0.6, 0, 0])
    top_bar = t1 + torch.linspace(0, 1, 10).reshape(-1, 1) * (t2 - t1)
    t3 = t1 + 0.5 * (t2 - t1)
    t4 = t3 + torch.tensor([0, 0, -0.65, 0])
    middle_bar = t3 + torch.linspace(0, 1, 20).reshape(-1, 1) * (t4 - t3)
    bottom_bar = top_bar - torch.tensor([0, 0, 0.65, 0])
    obs = torch.vstack((top_bar, middle_bar, bottom_bar))
    #obs = torch.vstack((middle_bar, bottom_bar))
    #obs = middle_bar

    n_dummy = 1
    dummy_obs = torch.hstack((torch.zeros(n_dummy, 3)+10, torch.zeros(n_dummy, 1)+0.1)).to(**params)
    obs = torch.vstack((obs, dummy_obs)).to(**params)

    # Integration parameters
    A = -1 * torch.diag(torch.ones(DOF)).to(**params)
    N_traj = config['integrator']['n_trajectories']
    dt_H = config['integrator']['horizon']
    dt_sim = config['integrator']['dt']
    N_ITER = 0

    #set up second mppi to move the robot
    mppi_step = MPPI(q_0, q_f, dh_params, obs, dt_sim, dt_H, N_traj, A, dh_a, nn_model)
    mppi_step.dst_thr = config['planner']['collision_threshold']   # subtracted from actual distance (added threshsold)
    mppi_step.Policy.alpha_s *= 0

    # ########################################
    # ###     GYM AND SIMULATION SETUP     ###
    # ########################################
    world_instance, robot_sim, robot_ptr, env_ptr = deploy_world_robot(gym_instance, params)
    w_T_r = copy.deepcopy(robot_sim.spawn_robot_pose)
    R_tens = getTensTransform(w_T_r)

    obs_list = []
    for i, sphere in enumerate(obs):
        tmpObsDict = deploy_sphere(sphere, gym_instance, w_T_r, 'sphere_%d'%(i), params)
        obs_list.append(tmpObsDict)

    ########################################
    ###     RUN MPPI AND SIMULATE        ###
    ########################################
    # warmup gym and jit
    for i in range(3):
        mppi_step.Policy.sample_policy()
        _, _, _ = mppi_step.propagate()
        gym_instance.step()
        robot_sim.set_robot_state(mppi_step.q_cur, mppi_step.q_cur*0, env_ptr, robot_ptr)
        numeric_fk_model(mppi_step.q_cur, dh_params, 10)
    best_idx = -1
    print('Init time: %4.2fs' % (time.time() - t00))
    time.sleep(1)
    t0 = time.time()
    all_fk_kernel = []
    while torch.norm(mppi_step.q_cur - q_f)+1 > 0.001:
        t_iter = time.time()

        # [ZMQ] Receive policy from planner
        policy_data = zmq_try_recv(policy_data, socket_receive_policy)
        mppi_step.Policy.update_with_data(policy_data)

        # [ZMQ] Receive obstacles
        mppi_step.obs = zmq_try_recv(mppi_step.obs, socket_receive_obs)

        # calculate FK for all kernels
        if len(all_fk_kernel) != mppi_step.Policy.n_kernels:
            all_fk_kernel = []
            for mu in mppi_step.Policy.mu_c[0:mppi_step.Policy.n_kernels]:
                kernel_fk, _ = numeric_fk_model(mu, dh_params, 3)
                fk_arr = kernel_fk.flatten(0, 1) @ R_tens[0:3, 0:3] + R_tens[0:3, 3]
                all_fk_kernel.append(fk_arr)

        # Propagate modulated DS


        gym_instance.clear_lines()
        # Update current robot state

        # Update current robot state
        mppi_step.Policy.sample_policy()    # samples a new policy using planned means and sigmas
        _, _, _ = mppi_step.propagate()
        mppi_step.q_cur = mppi_step.q_cur + mppi_step.qdot[0, :] * dt_sim

        # [ZMQ] Send current state to planner
        socket_send_state.send_pyobj(mppi_step.q_cur)

        goal_fk, _ = numeric_fk_model(q_f, dh_params, 2)
        # # draw lines in gym
        # draw kernels
        for fk in all_fk_kernel:
            gym_instance.draw_lines(fk, color=[1, 1, 1])
        gym_instance.draw_lines(goal_fk.flatten(0, 1) @ R_tens[0:3, 0:3] + R_tens[0:3, 3], color=[0.6, 1, 0.6])


        if N_ITER % 1 == 0:
            gym_instance.step()
        q_des = mppi_step.q_cur
        dq_des = mppi_step.qdot[0, :] * 0
        robot_sim.set_robot_state(q_des, dq_des, env_ptr, robot_ptr)

        N_ITER += 1
        if N_ITER > 10000:
            break
        # print(q_cur)
        t_iter = time.time() - t_iter
        time.sleep(max(0.0, 1/config['integrator']['desired_frequency'] - t_iter))
        print(f'Iteration:{N_ITER:4d}, Time:{t_iter:4.2f}, Frequency:{1/t_iter:4.2f},',
              f' Avg. frequency:{N_ITER/(time.time()-t0):4.2f}',
              f' Kernel count:{mppi_step.Policy.n_kernels:4d}')
        #print('Position difference: %4.3f'% (mppi_step.q_cur - q_f).norm().cpu())
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

