import sys, yaml
import zmq
sys.path.append('../functions/')
from MPPI import *
from zmq_utils import *
import torch

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
    # socket to publish policy from integrator loop
    socket_send_policy = init_publisher(context, '*', config["zmq"]["policy_port"])

    # socket to receive state from integrator loop
    socket_receive_state = init_subscriber(context, 'localhost', config["zmq"]["state_port"])

    # socket to receive obstacles
    socket_receive_obs = init_subscriber(context, 'localhost', config["zmq"]["obstacle_port"])

    ########################################
    ###        CONTROLLER SETUP          ###
    ########################################
    t00 = time.time()
    DOF = 7
    L = 1

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
    N_traj = config['planner']['n_trajectories']
    dt_H = config['planner']['horizon']
    dt = config['planner']['dt']
    N_ITER = 0

    # kernel adding thresholds
    dst_thr = config['planner']['kernel_adding_collision_thr']       # distance to collision (everything below - adds a kernel)
    thr_rbf_add = config['planner']['kernel_adding_kernels_thr']   # distance to closest kernel (l2 norm of 7d vector difference)

    # [ZMQ] Receive obstacles
    obs = zmq_init_recv(socket_receive_obs)
    #primary MPPI to sample naviagtion policy
    mppi = MPPI(q_0, q_f, dh_params, obs, dt, dt_H, N_traj, A, dh_a, nn_model)
    mppi.Policy.sigma_c_nominal = config['planner']['kernel_width']
    mppi.Policy.alpha_s = config['planner']['alpha_sampling_sigma']
    mppi.Policy.policy_upd_rate = config['planner']['policy_update_rate']
    mppi.dst_thr = dst_thr*0.5                                    # subtracted from actual distance (added threshsold)
    mppi.ker_thr = config['planner']['kernel_update_threshold']   # used to create update mask for policy means


    ########################################
    ###     RUN MPPI AND SIMULATE        ###
    ########################################

    print('Init time: %4.2fs' % (time.time() - t00))
    time.sleep(1)
    t0 = time.time()
    while torch.norm(mppi.q_cur - q_f)+1 > 0.001:
        t_iter = time.time()
        # [ZMQ] Receive state from integrator
        mppi.q_cur = zmq_try_recv(mppi.q_cur, socket_receive_state)
        # [ZMQ] Receive obstacles
        mppi.obs = zmq_try_recv(mppi.obs, socket_receive_obs)

        # Sample random policies
        mppi.Policy.sample_policy()
        # Propagate modulated DS
        # print(f'Init state: {mppi.q_cur}')
        with record_function("TAG: general propagation"):
            all_traj, closests_dist_all, kernel_val_all = mppi.propagate()

        with record_function("TAG: cost calculation"):
            # Calculate cost
            cost = mppi.get_cost() # don't delete, writes to self.cost
            mppi.shift_policy_means()

        # Check trajectory for new kernel candidates and add policy kernels
        kernel_candidates = mppi.Policy.check_traj_for_kernels(all_traj, closests_dist_all, dst_thr, thr_rbf_add)

        if len(kernel_candidates) > 0:
            rand_idx = torch.randint(kernel_candidates.shape[0], (1,))
            mppi.Policy.add_kernel(kernel_candidates[rand_idx[0]])
            kernel_fk, _ = numeric_fk_model(kernel_candidates[rand_idx[0]], dh_params, 3)

        # [ZMQ] Send current policy to integrator
        data = [mppi.Policy.n_kernels,
                mppi.Policy.mu_c[0:mppi.Policy.n_kernels],
                mppi.Policy.alpha_c[0:mppi.Policy.n_kernels],
                mppi.Policy.sigma_c[0:mppi.Policy.n_kernels]]
        socket_send_policy.send_pyobj(data)


        N_ITER += 1
        if N_ITER > 10000:
            break
        # print(q_cur)
        t_iter = time.time() - t_iter
        print(f'Iteration:{N_ITER:4d}, Time:{t_iter:4.2f}, Frequency:{1/t_iter:4.2f},',
              f' Avg. frequency:{N_ITER/(time.time()-t0):4.2f}',
              f' Kernel count:{mppi.Policy.n_kernels:4d}')
        #print('Position difference: %4.3f'% (mppi.q_cur - q_f).norm().cpu())
    td = time.time() - t0
    print('Time: ', td)
    time.sleep(10)


if __name__ == '__main__':
    main_loop()


