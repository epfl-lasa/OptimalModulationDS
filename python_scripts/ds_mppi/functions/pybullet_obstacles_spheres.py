import time
import numpy as np
import math


class SphereManager:
    def __init__(self, pybullet_client):
        self.pb = pybullet_client
        self.spheres = []
        self.color = [.7, .1, .1, .9]

    def create_sphere(self, position, radius, color):
        sphere = self.pb.createVisualShape(self.pb.GEOM_SPHERE,
                                           radius=radius,
                                           rgbaColor=color,
                                           specularColor=[0.4, .4, 0])
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

