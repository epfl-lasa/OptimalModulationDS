import time
import numpy as np
import math


class SphereManager:
    def __init__(self, pybullet_client):
        self.pb = pybullet_client
        self.spheres = []
        self.color = [.7, .1, .1, 1]

    def create_sphere(self, position, radius, color):
        sphere = self.pb.createVisualShape(self.pb.GEOM_SPHERE,
                                           radius=radius,
                                           rgbaColor=color)
        sphere = self.pb.createMultiBody(baseVisualShapeIndex=sphere,
                                         basePosition=position)
        self.spheres.append(sphere)

    def initialize_spheres(self, obstacle_array):
        for obstacle in obstacle_array:
            self.create_sphere(obstacle[0:3], obstacle[3], self.color)

    def delete_spheres(self):
        for sphere in self.spheres:
            self.pb.removeBody(sphere)
        self.spheres = []

    def update_spheres(self, obstacle_array):
        if (obstacle_array is not None) and (len(self.spheres) == len(obstacle_array)):
            for i, sphere in enumerate(self.spheres):
                self.pb.resetBasePositionAndOrientation(sphere,
                                                        obstacle_array[i, 0:3],
                                                        [1, 0, 0, 0])
        else:
            print("Number of spheres and obstacles do not match")
            self.delete_spheres()
            self.initialize_spheres(obstacle_array)


class KernelManager:
    def __init__(self, pybullet_client):
        self.pb = pybullet_client
        self.kernels = []
        self.color = [0, 1, 1]
        self.width = 2

    def create_line(self, positions):
        #draw a line
        tmp_arr = []
        for i in range(len(positions)-1):
            pos_current = positions[i]
            pos_next = positions[i+1]
            lineId = self.pb.addUserDebugLine(pos_current, pos_next, self.color, self.width, lifeTime=0)
            tmp_arr.append(lineId)
        self.kernels.append(tmp_arr)

    def initialize_kernels(self, kernel_array):
        for kernel in kernel_array:
            self.create_line(kernel)

    def delete_lines(self, line_arr):
        for line in line_arr:
            self.pb.removeUserDebugItem(line)

    def delete_kernels(self):
        self.pb.removeAllUserDebugItems()
        # for kernel in self.kernels:
        #     self.delete_lines(kernel)
        self.kernels = []

    def update_kernels(self, policy_data, key='kernel_fk'):
        if policy_data is not None:
            kernel_array = policy_data[key]
            if len(self.kernels) < len(kernel_array):
                for i in range(len(self.kernels), len(kernel_array)):
                    self.create_line(kernel_array[i])
            elif len(kernel_array) == 0:
                self.delete_kernels()
            elif len(self.kernels) > len(kernel_array):
                print("Number of spheres and obstacles do not match")
                self.delete_kernels()
                self.initialize_kernels(kernel_array)
        else:
            pass

# class StreamChecker:
#     def __init__(self, pybullet_client):
#         self.pb = pybullet_client
#         self.obs_stream = []
#         self.kernel_stream = []
#         self.state_stream = []
