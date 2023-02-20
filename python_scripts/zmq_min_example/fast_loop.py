import time
import torch
import zmq


## ZMQ setup
context = zmq.Context()
# socket to receive data from slow loop
socket_receive = context.socket(zmq.SUB)
socket_receive.setsockopt(zmq.CONFLATE, 1)
socket_receive.connect("tcp://localhost:1337")
socket_receive.setsockopt(zmq.SUBSCRIBE, b"")

# socket to publish data to slow loop
socket_send = context.socket(zmq.PUB)
socket_send.bind("tcp://*:1338")

t_init = time.time()
iter = 0
data = torch.zeros(2, 7)
while True:
    ## Iteration Start
    t_start = time.time()
    iter += 1

    ## Iteration Body
    #  Get the input from slow loop.
    try:
        data = socket_receive.recv_pyobj(zmq.DONTWAIT)
        print(f"Received reply {iter} \n {data}")
        print(info)
    except:
        pass
    # assume some work done (i.e. 1-step integration using data)
    time.sleep(1/500)  # work is 500 Hz
    data[1, 0] = iter
    #now send updated state to slow loop
    socket_send.send_pyobj(data)

    ## Iteration End
    t_end = time.time()
    t_iter = t_end - t_start
    t_all = t_end - t_init
    freq_iter = 1/t_iter
    freq_all = iter/t_all
    info = "Iterations: %d, Time: %4.2fs, Frequency: %4.2fHz, Avg. frequency: %4.2fHz" % (iter, t_all, freq_iter, freq_all)
    #print(info)

