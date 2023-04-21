import sys
sys.path.append('functions/')
from MPPI import *
import torch
from zmq_utils import *

def main_loop():
    # socket to publish data to slow loop
    context = zmq.Context()
    socket_send = init_publisher(context, '*', 1336)
    while True:
        dct = {'q':np.zeros(3),'dq':np.zeros(3)}
        socket_send.send_pyobj(dct)
        # time.sleep(0.1)
        print('hi')

if __name__ == '__main__':
    main_loop()

