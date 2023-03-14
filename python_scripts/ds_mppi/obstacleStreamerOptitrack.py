import yaml, sys
import torch
import numpy as np
sys.path.append('functions/')
from zmq_utils import *
from optitrack_utils import *

def read_yaml(fname):
    with open(fname) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

# define tensor parameters (cpu or cuda:0 or mps)
params = {'device': 'cpu', 'dtype': torch.float32}


def get_ball_pos(bodies, radius):
    for body in bodies:
        if body['id'] == 1001:
            return torch.hstack([body['pos'], torch.tensor(radius)])
    return torch.tensor([10, 10, 10, torch.tensor(radius)])

def main_loop():

    config = read_yaml('config.yaml')

    ########################################
    ###            ZMQ SETUP             ###
    ########################################
    context = zmq.Context()
    # socket to publish obstacles
    socket_send_obs = init_publisher(context, '*', config["zmq"]["obstacle_port"])

    # socket to receive state from integrator
    socket_receive_optitrack = init_subscriber(context, '128.178.145.79', 5511)

    freq = config["obstacle_streamer"]["frequency"]
    t_0 = time.time()
    N_ITER = 0
    state = None
    # create square pointcloud at table base

    n_dummy = 50
    obs = torch.hstack((torch.zeros(n_dummy, 3) + 10, torch.zeros(n_dummy, 1) + 0.05)).to(**params)
    while True:
        moved = False
        t_run = time.time() - t_0

        # get robot state
        optitrack_data, optitrack_recv_status = zmq_try_recv_raw(None, socket_receive_optitrack)

        if optitrack_recv_status:
            bodies = process_raw_message(optitrack_data)
            #print(bodies)
            if len(bodies) > 0:
                obs[0] = get_ball_pos(bodies, 0.05)
        socket_send_obs.send_pyobj(obs)
        time.sleep(1/freq)
        N_ITER += 1
        if N_ITER % freq == 0:
            print(f"Streaming at {freq} Hz for {int(t_run):d} seconds.")

if __name__ == '__main__':
    main_loop()
