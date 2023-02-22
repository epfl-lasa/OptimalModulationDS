import yaml
import zmq
import torch
import time


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


    t1 = torch.tensor([0.4, -0.3, 0.75, .05])
    t2 = t1 + torch.tensor([0, 0.6, 0, 0])
    top_bar = t1 + torch.linspace(0, 1, 10).reshape(-1, 1) * (t2 - t1)
    t3 = t1 + 0.5 * (t2 - t1)
    t4 = t3 + torch.tensor([0, 0, -0.65, 0])
    middle_bar = t3 + torch.linspace(0, 1, 20).reshape(-1, 1) * (t4 - t3)
    bottom_bar = top_bar - torch.tensor([0, 0, 0.65, 0])
    obs = torch.vstack((top_bar, middle_bar, bottom_bar))
    # obs = torch.vstack((middle_bar, bottom_bar))
    n_dummy = 1
    dummy_obs = torch.hstack((torch.zeros(n_dummy, 3) + 10, torch.zeros(n_dummy, 1) + 0.1)).to(**params)
    obs = torch.vstack((obs, dummy_obs)).to(**params)

    i = 0
    freq = config["obstacle_streamer"]["frequency"]
    while True:
        socket_send_obs.send_pyobj(obs)
        time.sleep(1/freq)
        i += 1
        if i % freq == 0:
            print(f"Streaming at {freq} Hz for {i/120} seconds.")

if __name__ == '__main__':
    main_loop()
