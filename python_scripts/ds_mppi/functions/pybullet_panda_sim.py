import time
import numpy as np
from math import pi


pandaNumDofs = 7

# restpose
rp = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]

class PandaSim():
    def __init__(self, bullet_client, base_pos, base_rot):
        self.bullet_client = bullet_client
        self.bullet_client.setAdditionalSearchPath('content/urdfs')
        self.base_pos = np.array(base_pos)
        self.base_rot = np.array(base_rot)
        # print("offset=",offset)
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        table_rot = self.bullet_client.getQuaternionFromEuler([pi / 2, 0, pi])
        # self.bullet_client.loadURDF('LAB/lab.urdf', np.array([1.01, -0.28, 0.45]), table_rot, flags=flags)
        #self.bullet_client.loadURDF('lab_table/table.urdf', np.array([-0.15, 0.02, -0.1]), table_rot, flags=flags)
        # self.bullet_client.loadURDF('plane.urdf', np.array([0, 0, 0]), np.array([0, 0, 0, 1]), flags=flags)

        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", self.base_pos,
                                                 self.base_rot, useFixedBase=True, flags=flags)
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.panda, j, rp[index])
                index = index + 1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.panda, j, rp[index])
                index = index + 1
        self.t = 0.
        self.set_joint_positions(rp)

    def reset(self):
        pass

    def set_joint_positions(self, joint_positions):
        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     joint_positions[i], force=5 * 240.)
        self.set_finger_positions(0.04)

    def set_finger_positions(self, gripper_opening):
        self.bullet_client.setJointMotorControl2(self.panda, 9, self.bullet_client.POSITION_CONTROL,
                                                 gripper_opening/2, force=5 * 240.)
        self.bullet_client.setJointMotorControl2(self.panda, 10, self.bullet_client.POSITION_CONTROL,
                                                 -gripper_opening/2, force=5 * 240.)


    def get_joint_positions(self):
        joint_state = []
        for i in range(pandaNumDofs):
            joint_state.append(self.bullet_client.getJointState(self.panda, i)[0])
        return joint_state

