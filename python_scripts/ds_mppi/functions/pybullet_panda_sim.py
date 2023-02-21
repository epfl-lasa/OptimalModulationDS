import time
import numpy as np
import math

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11  # 8
pandaNumDofs = 7

ll = [-7] * pandaNumDofs
# upper limits for null space (todo: set them to proper range)
ul = [7] * pandaNumDofs
# joint ranges for null space (todo: set them to proper range)
jr = [7] * pandaNumDofs
# restposes for null space
jointPositions = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions


class PandaSim():
    def __init__(self, bullet_client, base_pos, base_rot):
        self.bullet_client = bullet_client
        bullet_client.setAdditionalSearchPath('content/urdfs/')
        self.base_pos = np.array(base_pos)
        self.base_rot = np.array(base_rot)
        # print("offset=",offset)
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        table_rot = self.bullet_client.getQuaternionFromEuler([math.pi / 2, 0, math.pi])
        self.bullet_client.loadURDF('LAB/lab.urdf', np.array([1.03, -0.3, 0.45]), table_rot, flags=flags)
        # self.bullet_client.loadURDF('lab_table/table.urdf', np.array([-0.15, 0.02, -0.1]), table_rot, flags=flags)
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
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
        self.t = 0.
        self.set_joint_positions(rp)

    def reset(self):
        pass

    def set_joint_positions(self, joint_positions):
        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     joint_positions[i], force=5 * 240.)
