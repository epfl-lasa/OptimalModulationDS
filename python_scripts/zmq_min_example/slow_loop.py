import time
import torch
import zmq


## ZMQ setup
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:1337")

t_init = time.time()
iter = 0
tensor_size = [2, 7]
while True:
    ## Iteration Start
    t_start = time.time()
    iter += 1
    ## Iteration Body
    time.sleep(1/10)
    data = iter * torch.ones(tensor_size)
    socket.send_pyobj(data)
    ## Iteration End
    t_end = time.time()
    t_iter = t_end - t_start
    t_all = t_end - t_init
    freq_iter = 1/t_iter
    freq_all = iter/t_all
    info = "Iterations: %d, Time: %4.2fs, Frequency: %4.2fHz, Avg. frequency: %4.2fHz" % (iter, t_all, freq_iter, freq_all)
    print(info)

