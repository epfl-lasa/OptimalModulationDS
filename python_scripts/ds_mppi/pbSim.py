import pybullet as p
import pybullet_data as pd
import math
import numpy as np
import sys
import time
from zmq_utils import *
import yaml

def main_loop():
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    sys.path.append('functions/')
    from pybullet_panda_sim import PandaSim

    p.connect(p.GUI, options='--background_color_red=0.2 --background_color_green=0.2' +
                             ' --background_color_blue=0.2 --width=1600 --height=1000')
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(lightPosition=[20, 0, 100])
    p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0, 0, 0.5])
    p.setAdditionalSearchPath(pd.getDataPath())
    timeStep = 0.01
    p.setTimeStep(timeStep)
    p.setGravity(0, 0, -9.81)

    ## spawn franka robot
    base_pos = [0, 0, 0]
    base_rot = p.getQuaternionFromEuler([0, 0, 0])
    panda = PandaSim(p, base_pos, base_rot)

    ########################################
    ###            ZMQ SETUP             ###
    ########################################
    context = zmq.Context()

    # socket to receive state from integrator
    socket_receive_state = init_subscriber(context, 'localhost', config["zmq"]["state_port"])

    # socket to receive obstacles
    socket_receive_obs = init_subscriber(context, 'localhost', config["zmq"]["obstacle_port"])
    desired_frequency = config["simulator"]["max_frequency"]
    #state = [-1.2, -0.08, 0, -2, -0.16,  1.6, -0.75]
    state = config["general"]["q_0"]
    obs = None
    ## main loop
    while True:
        t_iter_begin = time.time()
        # [ZMQ] Receive obstacles
        obs = zmq_try_recv(obs, socket_receive_obs)

        # [ZMQ] Receive state from integrator
        state = zmq_try_recv(state, socket_receive_state)
        panda.set_joint_positions(state)
        p.stepSimulation()
        t_iter_end = time.time()
        t_iter = t_iter_end - t_iter_begin
        time.sleep(max(t_iter-1/desired_frequency, 0))


if __name__ == '__main__':
    main_loop()
