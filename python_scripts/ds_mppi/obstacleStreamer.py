import yaml, sys
import torch
import time
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
    ########################################
    ### I-Shape centered in front of franka
    ########################################
    x_dist = 0.4
    y_width = 0.4
    z_0 = 0.1
    height = 0.75
    r = 0.05
    n_vertical = 20
    n_horizontal = 20

    # # small one
    # z_0 = 0.35
    # height = 0.4
    # y_width = 0.2
    # x_dist = 0.35

    top_left = torch.tensor([x_dist, -y_width, z_0+height, r])
    top_right = torch.tensor([x_dist, y_width, z_0+height, r])
    top_bar = top_left + torch.linspace(0, 1, n_horizontal).reshape(-1, 1) * (top_right - top_left)
    bottom_bar = top_bar - torch.tensor([0, 0, height, 0])

    top_mid = top_left + 0.5 * (top_right - top_left)
    bottom_mid = top_mid - torch.tensor([0, 0, height, 0])
    middle_bar = bottom_mid + torch.linspace(0, 1, n_vertical).reshape(-1, 1) * (top_mid - bottom_mid)
    tshape = torch.vstack((top_bar, middle_bar, bottom_bar))
    ########################################
    ### ring constrained
    ########################################

    center = torch.tensor([.55, 0, 0.6])
    radius = 0.2
    n_ring = 21
    ring = torch.zeros(n_ring, 4)
    ring[:, 0] = center[0]
    ring[:, 1] = center[1] + radius * torch.cos(torch.linspace(0, 2 * np.pi, n_ring))
    ring[:, 2] = center[2] + radius * torch.sin(torch.linspace(0, 2 * np.pi, n_ring))
    ring[:, 3] = 0.03

    ########################################
    ### line/wall
    ########################################
    r = 0.05
    n_pts = 2
    length = max(1, 2*n_pts-2)*r
    z0 = 0.5
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

    ########################################
    ### shelf
    ########################################
    r = 0.05
    n_pts = 8
    length = max(1, 2*n_pts-2)*r
    z0 = 0.25
    x0 = 0.4
    y0 = 0
    posA = torch.tensor([x0, y0, z0+length, r])
    posB = posA + torch.tensor([length, 0.0, 0.0, 0.0])
    line = posA + torch.linspace(0, 1, n_pts).reshape(-1, 1) * (posB - posA)
    shelf = line
    for sphere in line:
        sphere_down = sphere - torch.tensor([0, 0, length, 0])
        line_down = sphere + torch.linspace(0, 1, n_pts).reshape(-1, 1) * (sphere_down - sphere)
        sphere_left = sphere + torch.tensor([0, -length/2, -length/2, 0])
        sphere_right = sphere + torch.tensor([0, length/2, -length/2, 0])
        line_lr = sphere_left + torch.linspace(0, 1, n_pts).reshape(-1, 1) * (sphere_right - sphere_left)
        shelf = torch.vstack((shelf, line_down, line_lr))

    ########################################
    ### Dummy obstacle
    ########################################
    # n_dummy = 1
    # dummy_obs = torch.hstack((torch.zeros(n_dummy, 3) + 10, torch.zeros(n_dummy, 1) + 0.1)).to(**params)
    # obs = torch.vstack((obs, dummy_obs)).to(**params)

    N_ITER = 0
    freq = config["obstacle_streamer"]["frequency"]
    amplitude_array = torch.tensor([[0.0, 0.0, 0.0, 0],
                                    [0.0, 0.0, 0.0, 0]])
    period_array = [40, 0.01]
    t_0 = time.time()
    while True:
        config = read_yaml('config.yaml')
        if config["collision_model"]["obstacle"] == 'ring':
            obs = ring
        elif config["collision_model"]["obstacle"] == 'tshape':
            obs = tshape
        elif config["collision_model"]["obstacle"] == 'wall':
            obs = wall
        elif config["collision_model"]["obstacle"] == 'line':
            obs = line
        elif config["collision_model"]["obstacle"] == 'shelf':
            obs = shelf
        else:
            obs = torch.tensor([[10.3, 0.0, 0.0, 0.01]])
        t_run = time.time() - t_0
        delta = obs * 0
        for i in range(len(amplitude_array)):
            delta += amplitude_array[i] * np.sin(2 * np.pi * t_run / period_array[i])
        socket_send_obs.send_pyobj(obs+delta)
        time.sleep(1/freq)
        N_ITER += 1
        if N_ITER % freq == 0:
            print(f"Streaming at {freq} Hz for {int(t_run):d} seconds.")

if __name__ == '__main__':
    main_loop()
