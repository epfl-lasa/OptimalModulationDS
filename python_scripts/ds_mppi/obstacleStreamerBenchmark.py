import yaml, sys
import torch
import numpy as np
sys.path.append('functions/')
from zmq_utils import *


def read_yaml(fname):
    with open(fname) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

# define tensor parameters (cpu or cuda:0 or mps)
params = {'device': 'cpu', 'dtype': torch.float32}
def main_loop():

    config = read_yaml('config.yaml')

    ########################################
    ###            ZMQ SETUP             ###
    ########################################
    context = zmq.Context()
    # socket to publish obstacles
    socket_send_obs = init_publisher(context, '*', config["zmq"]["obstacle_port"])

    # socket to receive state from integrator
    socket_receive_state = init_subscriber(context, 'localhost', config["zmq"]["state_port"])


    ########################################
    ### line/wall
    ########################################
    r = 0.05
    n_pts = 4
    length = max(1, 2*n_pts-2)*r
    z0 = 0.8
    x0 = 0.6
    y0 = 0
    posA = torch.tensor([x0, y0, z0+length, r])
    posB = posA + torch.tensor([length, 0.0, 0.0, 0.0])
    line = posA + torch.linspace(0, 1, n_pts).reshape(-1, 1) * (posB - posA)
    wall = line
    n_down = n_pts
    for sphere in line:
        sphere_down = sphere - torch.tensor([0, 0, length, 0])
        line_down = sphere + torch.linspace(0, 1, n_down).reshape(-1, 1) * (sphere_down - sphere)
        wall = torch.vstack((wall, line_down))
    obs = wall

    ########################################
    ### cross
    ########################################
    r = 0.05
    n_pts = 3
    z0 = 0.9
    x0 = 0.3
    y0 = 0
    length = max(1, 2*n_pts - 2) * r
    center = torch.tensor([x0, y0, z0, r])
    top = center + torch.tensor([0, 0, length, 0])
    bottom = center + torch.tensor([0, 0, -length, 0])
    left = center + torch.tensor([0, -length, 0, 0])
    right = center + torch.tensor([0, length, 0, 0])
    line_h = left + torch.linspace(0, 1, 2*n_pts).reshape(-1, 1) * (right - left)
    line_v = top + torch.linspace(0, 1, 2*n_pts).reshape(-1, 1) * (bottom - top)
    cross = torch.vstack((line_v, line_h))
    obs = cross

    thr_bottom = 0.3
    x_shift = 0.1
    N_ITER = 0
    freq = config["obstacle_streamer"]["frequency"]
    t_0 = time.time()

    q_0 = torch.tensor(config["general"]["q_0"])
    state = None
    while True:
        moved = False
        t_run = time.time() - t_0

        # get robot state
        state, state_recv_status = zmq_try_recv(state, socket_receive_state)

        if state_recv_status and (state - q_0).norm().numpy() < 1e-6:
            obs -= torch.tensor([0, 0, 0.05, 0])
            moved = True
        # send obstacles
        socket_send_obs.send_pyobj(obs)

        time.sleep(1/freq)
        if moved:
            print(obs[0])
            if obs[0][2] < thr_bottom:
                hdif_init = z0 + length - obs[0][2]
                obs += torch.tensor([0, 0, hdif_init, 0])
                obs += torch.tensor([x_shift, 0, 0, 0])
            time.sleep(1.1)

        N_ITER += 1
        if N_ITER % freq == 0:
            print(f"Streaming at {freq} Hz for {int(t_run):d} seconds.")

if __name__ == '__main__':
    main_loop()
