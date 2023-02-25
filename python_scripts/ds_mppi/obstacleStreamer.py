import yaml
import zmq
import torch
import time
import numpy as np


# define tensor parameters (cpu or cuda:0 or mps)
params = {'device': 'cpu', 'dtype': torch.float32}
def main_loop():

    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ########################################
    ###            ZMQ SETUP             ###
    ########################################
    context = zmq.Context()
    # socket to publish data to fast loop
    socket_send_obs = context.socket(zmq.PUB)
    socket_send_obs.bind("tcp://*:%d" % config["zmq"]["obstacle_port"])


    ### I-Shape centered in front of franka
    x_dist = 0.4
    y_width = 0.4
    z_0 = 0.1
    height = 0.75
    r = 0.05
    n_vertical = 20
    n_horizontal = 20

    top_left = torch.tensor([x_dist, -y_width, z_0+height, r])
    top_right = torch.tensor([x_dist, y_width, z_0+height, r])
    top_bar = top_left + torch.linspace(0, 1, n_horizontal).reshape(-1, 1) * (top_right - top_left)
    bottom_bar = top_bar - torch.tensor([0, 0, height, 0])

    top_mid = top_left + 0.5 * (top_right - top_left)
    bottom_mid = top_mid - torch.tensor([0, 0, height, 0])
    middle_bar = bottom_mid + torch.linspace(0, 1, n_vertical).reshape(-1, 1) * (top_mid - bottom_mid)
    obs = torch.vstack((top_bar, middle_bar, bottom_bar))
    # obs = torch.vstack((middle_bar, bottom_bar))
    n_dummy = 1
    dummy_obs = torch.hstack((torch.zeros(n_dummy, 3) + 10, torch.zeros(n_dummy, 1) + 0.1)).to(**params)
    obs = torch.vstack((obs, dummy_obs)).to(**params)

    N_ITER = 0
    freq = config["obstacle_streamer"]["frequency"]
    amplitude_array = torch.tensor([[0, 0, 0.0, 0],
                                    [0.01, 0, 0, 0]])
    period_array = [2, 2]
    t_0 = time.time()
    while True:
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
