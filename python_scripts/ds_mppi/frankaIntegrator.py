import sys
sys.path.append('functions/')
from MPPI import *
import torch
from zmq_utils import *
import yaml

sys.path.append('../mlp_learn/')
from sdf.robot_sdf import RobotSdfCollisionNet

# define tensor parameters (cpu or cuda:0 or mps)
params = {'device': 'cpu', 'dtype': torch.float32}

def main_loop():

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

    # Integration parameters
    A = -1 * torch.diag(torch.ones(DOF)).to(**params)
    N_traj = config['integrator']['n_trajectories']
    dt_H = config['integrator']['horizon']
    dt_sim = config['integrator']['dt']
    N_ITER = 0

    #set up one-step one-trajectory mppi to move the robot
    mppi_step = MPPI(q_0, q_f, dh_params, obs, dt_sim, dt_H, N_traj, A, dh_a, nn_model)
    mppi_step.dst_thr = config['integrator']['collision_threshold']   # subtracted from actual distance (added threshsold)
    mppi_step.Policy.alpha_s *= 0


    ########################################
    ###     RUN MPPI AND SIMULATE        ###
    ########################################
    for i in range(20):
        socket_send_state.send_pyobj(q_0)
        time.sleep(0.1)
    print('Init time: %4.2fs' % (time.time() - t00))
    des_freq = config['integrator']['desired_frequency']
    t0 = time.time()
    all_fk_kernel = []
    while torch.norm(mppi_step.q_cur - q_f)+1 > 0.001:
        t_iter_start = time.time()

        # [ZMQ] Receive policy from planner
        policy_data, policy_recv_status = zmq_try_recv(policy_data, socket_receive_policy)
        mppi_step.Policy.update_with_data(policy_data)

        # [ZMQ] Receive obstacles
        mppi_step.obs, obs_recv_status = zmq_try_recv(mppi_step.obs, socket_receive_obs)

        # # calculate FK for all kernels
        # if len(all_fk_kernel) != mppi_step.Policy.n_kernels:
        #     all_fk_kernel = []
        #     for mu in mppi_step.Policy.mu_c[0:mppi_step.Policy.n_kernels]:
        #         kernel_fk, _ = numeric_fk_model(mu, dh_params, 3)
        #         fk_arr = kernel_fk.flatten(0, 1) @ R_tens[0:3, 0:3] + R_tens[0:3, 3]
        #         all_fk_kernel.append(fk_arr)

        # Propagate modulated DS
        mppi_step.Policy.sample_policy()    # samples a new policy using planned means and sigmas
        _, _, _ = mppi_step.propagate()

        # Update current robot state
        mppi_step.q_cur = mppi_step.q_cur + mppi_step.qdot[0, :] * dt_sim
        mppi_step.q_cur = torch.clamp(mppi_step.q_cur, mppi_step.Cost.q_min, mppi_step.Cost.q_max)
        # [ZMQ] Send current state to planner
        q_des = mppi_step.q_cur
        dq_des = mppi_step.qdot[0, :] * 0

        socket_send_state.send_pyobj(q_des)

        N_ITER += 1
        if torch.norm(mppi_step.q_cur - q_f) < 0.1:
            mppi_step.q_cur = q_0
            socket_send_state.send_pyobj(mppi_step.q_cur)
            time.sleep(1)
        # print(q_cur)

        t_iter_tmp = time.time() - t_iter_start
        time.sleep(max(0.0, 1/des_freq - t_iter_tmp))
        t_iter = time.time() - t_iter_start
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
    main_loop()

