import pybullet as p
import pybullet_data as pd
import numpy as np
import sys
import time
import yaml

sys.path.append('functions/')
from pybullet_panda_sim import PandaSim
from pybullet_extras import SphereManager, KernelManager
from zmq_utils import *


def main_loop():
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


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

    # socket to receive data from slow loop
    socket_receive_policy = init_subscriber(context, 'localhost', config["zmq"]["policy_port"])
    policy_data = None

    #state = [-1.2, -0.08, 0, -2, -0.16,  1.6, -0.75]
    state = config["general"]["q_0"]
    obs = []
    sphere_manager = SphereManager(p)
    kernel_manager = KernelManager(p)
    ## main loop
    i = 0
    no_ker_upd = 0
    desired_frequency = config["simulator"]["max_frequency"]
    while True:
        t_iter_begin = time.time()
        # [ZMQ] Receive obstacles
        obs, obs_recv_status = zmq_try_recv(obs, socket_receive_obs)
        sphere_manager.update_spheres(obs)

        # [ZMQ] Receive policy from planner
        policy_data, policy_recv_status = zmq_try_recv(policy_data, socket_receive_policy)
        kernel_manager.update_kernels(policy_data)
        if not policy_recv_status:
            no_ker_upd += 1
        else:
            no_ker_upd = 0
        if no_ker_upd > 3*desired_frequency: # if no policy update for 3 seconds, delete kernels
            kernel_manager.delete_kernels()
            no_ker_upd = 0
            policy_data = None
        # [ZMQ] Receive state from integrator
        state, state_recv_status = zmq_try_recv(state, socket_receive_state)
        panda.set_joint_positions(state)
        p.stepSimulation()
        t_iter_end = time.time()
        t_iter = t_iter_end - t_iter_begin
        time.sleep(max(1/desired_frequency - t_iter, 0))
        i += 1
        if i % desired_frequency == 0:
            print(f"Simulating at {desired_frequency} Hz for {int(i/desired_frequency)} seconds.")


if __name__ == '__main__':
    main_loop()
