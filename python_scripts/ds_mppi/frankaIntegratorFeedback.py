import sys
sys.path.append('functions/')
from MPPI import *
import torch
from zmq_utils import *
import yaml
from LinDS import *
from SEDS import *

sys.path.append('../mlp_learn/')
from sdf.robot_sdf import RobotSdfCollisionNet

# define tensor parameters (cpu or cuda:0 or mps)
params = {'device': 'cpu', 'dtype': torch.float32}

def main_loop():

    with open('config_real.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ########################################
    ###            ZMQ SETUP             ###
    ########################################
    context = zmq.Context()
    # socket to receive data from slow loop
    socket_receive_policy = init_subscriber(context, 'localhost', config["zmq"]["policy_port"])


    # socket to receive data from robot
    ip_pc = '128.178.145.79'
    #ip_pc = '128.178.145.5'
    socket_receive_robot = init_subscriber(context, ip_pc, 6969)

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

    # Integration parameters
    DS1 = LinDS(q_f)
    DS2 = LinDS(q_0)
    DS_ARRAY = [DS1, DS2]
    N_traj = config['integrator']['n_trajectories']
    dt_H = config['integrator']['horizon']
    dt_sim = config['integrator']['dt']
    N_ITER_TOTAL = 0
    N_ITER_TRAJ = 0
    N_SUCCESS = 0
    SLEEP_SUCCESS = 1
    #set up one-step one-trajectory mppi to move the robot
    n_closest_obs = config['collision_model']['closest_spheres']
    mppi_step = MPPI(q_0, q_f, dh_params, obs, dt_sim, dt_H, N_traj, DS_ARRAY, dh_a, nn_model, n_closest_obs)
    mppi_step.dst_thr = config['integrator']['collision_threshold']   # subtracted from actual distance (added threshsold)
    mppi_step.Policy.alpha_s *= 0

    ########################################
    ###     RUN MPPI AND SIMULATE        ###
    ########################################
    print('Init time: %4.2fs' % (time.time() - t00))
    des_freq = config['integrator']['desired_frequency']
    t0 = time.time()
    t_traj_start = t0

    init_pos_status = 0
    q_init = q_0
    while not init_pos_status:
        q_init, init_pos_status = zmq_try_recv(None, socket_receive_robot)
        time.sleep(0.01)
    mppi_step.q_cur = torch.tensor(q_init).to(**params)

    while True:
        t_iter_start = time.time()

        # [ZMQ] Recieve robot state
        robot_state, robot_recv_status = zmq_try_recv(mppi_step.q_cur, socket_receive_robot)
        if robot_recv_status:
            print(robot_state)
            robot_state = torch.tensor(robot_state).to(**params)
            if (robot_state is not None) and (robot_state - mppi_step.q_cur).norm(p=2, dim=-1) > 0.5:
                print('resetting state to robot')
                mppi_step.q_cur = robot_state
            # if (robot_state is not None):
            #     mppi_step.q_cur = robot_state

            # [ZMQ] Receive policy from planner
            policy_data, policy_recv_status = zmq_try_recv(policy_data, socket_receive_policy)
            mppi_step.Policy.update_with_data(policy_data)
            # [ZMQ] Receive obstacles
            obstacles_data, obs_recv_status = zmq_try_recv(mppi_step.obs, socket_receive_obs)
            mppi_step.update_obstacles(obstacles_data)
            # Propagate modulated DS
            mppi_step.Policy.sample_policy()    # samples a new policy using planned means and sigmas
            _, _, _, _ = mppi_step.propagate()
            # # Update current robot state
            # mppi_step.q_cur = mppi_step.q_cur + mppi_step.qdot[0, :] * dt_sim
            # mppi_step.q_cur = torch.clamp(mppi_step.q_cur, mppi_step.Cost.q_min, mppi_step.Cost.q_max)
            # # [ZMQ] Send current state to planner
            # q_des = mppi_step.q_cur
            # dq_des = mppi_step.qdot[0, :] * 0
            q_des = mppi_step.q_cur + mppi_step.qdot[0, :] * dt_sim
            dq_des = mppi_step.qdot[0, :]
            mppi_step.q_cur = q_des
            state_dict = {'q': q_des, 'dq': dq_des, 'ds_idx': mppi_step.DS_idx}
            socket_send_state.send_pyobj(state_dict)
            N_ITER_TOTAL += 1
            N_ITER_TRAJ += 1
            t_traj = time.time() - t_traj_start
            if (torch.norm(mppi_step.q_cur - mppi_step.qf) < 0.1) and (N_ITER_TRAJ > 100):
                print('Switching DS!!')
                mppi_step.switch_DS_idx(N_SUCCESS % 2)
                status = f'1: Goal reached in {N_ITER_TRAJ:3d} iterations ({t_traj:4.2f} seconds). ' \
                         f'Obs: {obstacles_data.shape[0]}, Top obs: {obstacles_data[0]}'
                print(status)
                N_ITER_TRAJ = 0
                N_SUCCESS += 1
                time.sleep(0.5)
                t_traj_start = time.time()

            t_iter = time.time() - t_iter_start
            np.set_printoptions(precision=2, suppress=True)
            # print(f'Iteration: {N_ITER_TRAJ:4d}({N_ITER_TOTAL:4d} total), Time:{t_iter:4.2f}, Frequency:{1/t_iter:4.2f},',
            #       f' Avg. frequency:{N_ITER_TOTAL/(time.time()-t0-N_SUCCESS*SLEEP_SUCCESS):4.2f}',
            #       f' Kernel count:{mppi_step.Policy.n_kernels:4d}',
            #       f'Distance to collision: {mppi_step.closest_dist_all[0,0]*100:4.2f}cm',
            #       f'Kernel weights: {mppi_step.ker_w.numpy()}')
            # print(mppi_step.Policy.alpha_c[0:mppi_step.Policy.n_kernels].numpy())
            #print('Position difference: %4.3f'% (mppi_step.q_cur - q_f).norm().cpu())


if __name__ == '__main__':
    main_loop()

