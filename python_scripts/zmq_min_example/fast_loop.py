import time
import torch
import zmq


## ZMQ setup
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.CONFLATE, 1)
socket.connect("tcp://localhost:1337")
socket.setsockopt(zmq.SUBSCRIBE, b"")

t_init = time.time()
iter = 0

while True:
    ## Iteration Start
    t_start = time.time()
    iter += 1
    ## Iteration Body
    #  Get the reply.
    try:
        data = socket.recv_pyobj(zmq.DONTWAIT)
        print(f"Received reply {iter} \n {data}")
        print(info)
    except:
        pass
    time.sleep(1/500)
    ## Iteration End
    t_end = time.time()
    t_iter = t_end - t_start
    t_all = t_end - t_init
    freq_iter = 1/t_iter
    freq_all = iter/t_all
    info = "Iterations: %d, Time: %4.2fs, Frequency: %4.2fHz, Avg. frequency: %4.2fHz" % (iter, t_all, freq_iter, freq_all)
    #print(info)

