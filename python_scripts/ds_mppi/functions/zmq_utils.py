import zmq
import time

def zmq_init_recv(socket):
    val = None
    while val is None:
        try:
            val = socket.recv_pyobj(flags=zmq.DONTWAIT)
        except:
            print('No input data! (yet) waiting...')
            time.sleep(0.1)
            pass
    return val


def zmq_try_recv(val, socket):
    try:
        val = socket.recv_pyobj(flags=zmq.DONTWAIT)
    except:
        pass
    return val

def init_subscriber(context, address, port):
    # socket to receive stuff
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect("tcp://%s:%s" % (address, str(port)))
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    return socket


def init_publisher(context, address, port):
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://%s:%s" % (address, str(port)))
    return socket

