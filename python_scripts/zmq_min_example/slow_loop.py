import time
import torch
import zmq


## ZMQ setup
# socket to publish data to fast loop
context = zmq.Context()
socket_send = context.socket(zmq.PUB)
socket_send.bind("tcp://*:1337")
# socket to receive data from fast loop
socket_receive = context.socket(zmq.SUB)
socket_receive.setsockopt(zmq.CONFLATE, 1)
socket_receive.connect("tcp://localhost:1338")
socket_receive.setsockopt(zmq.SUBSCRIBE, b"")

t_init = time.time()
iter = 0
tensor_size = [2, 7]
data = torch.zeros(tensor_size)

while True:
    ## Iteration Start
    t_start = time.time()
    iter += 1

    ## Iteration Body
    #  Get the input from fast loop.
    try:
        data = socket_receive.recv_pyobj(zmq.DONTWAIT)
        print(f"Received reply {iter} \n {data}")
        print(info)
    except:
        pass
    # assume some work done (i.e. path-planning for state in data)
    time.sleep(1/10) #  work is 10 Hz
    data[0, 0] = iter
    # now send updated data to the fast loop
    socket_send.send_pyobj(data)

    ## Iteration End
    t_end = time.time()
    t_iter = t_end - t_start
    t_all = t_end - t_init
    freq_iter = 1/t_iter
    freq_all = iter/t_all
    info = "Iterations: %d, Time: %4.2fs, Frequency: %4.2fHz, Avg. frequency: %4.2fHz" % (iter, t_all, freq_iter, freq_all)
    print(info)

