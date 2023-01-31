import copy, time, argparse, os, sys, math, numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array
from torch import pi

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import *
from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform

import torch

## create robot simulation
def deploy_world_robot(gym_instance, params):
    robot_yml = 'content/configs/gym/franka.yml'
    sim_params = load_yaml(robot_yml)['sim_params']
    sim_params['asset_root'] = os.path.dirname(__file__) + '/content/assets'
    sim_params['collision_model'] = None
    # task_file = 'franka_reacher_env2.yml'
    gym = gym_instance.gym
    sim = gym_instance.sim
    robot_sim = RobotSim(gym_instance=gym_instance.gym, sim_instance=gym_instance.sim, **sim_params,
                         device=params['device'])
    robot_pose = sim_params['robot_pose']
    env_ptr = gym_instance.env_list[0]
    robot_ptr = robot_sim.spawn_robot(env_ptr, robot_pose, coll_id=2)

    # create world
    world_yml = 'content/configs/gym/collision_demo.yml'
    world_params = load_yaml(world_yml)
    # get pose
    w_T_r = copy.deepcopy(robot_sim.spawn_robot_pose)

    w_T_robot = torch.eye(4)
    quat = torch.tensor([w_T_r.r.w, w_T_r.r.x, w_T_r.r.y, w_T_r.r.z]).unsqueeze(0)
    rot = quaternion_to_matrix(quat)
    w_T_robot[0, 3] = w_T_r.p.x
    w_T_robot[1, 3] = w_T_r.p.y
    w_T_robot[2, 3] = w_T_r.p.z
    w_T_robot[:3, :3] = rot[0]
    world_instance = World(gym, sim, env_ptr, world_params, w_T_r=w_T_r)
    return world_instance, robot_sim, robot_ptr, env_ptr

def deploy_sphere(sphere_data, gym_instance, w_T_r, sph_name, params):
    env_ptr = gym_instance.env_list[0]
    sphere_dict = dict()
    sphere_dict['position'] = sphere_data[0:3]
    sphere_dict['amplitude'] = 0
    sphere_dict['velocity'] = 0
    sphere_dict['radius'] = sphere_data[3]
    sphere_dict['pose'] = gymapi.Transform()
    sphere_dict['pose'].p = gymapi.Vec3(sphere_data[0], sphere_data[1], sphere_data[2])
    sphere_dict['pose'].r = gymapi.Quat(0, 0, 0, 1)
    sphere_dict['name'] = sph_name
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.002
    sphere_dict['obs_asset'] = gym_instance.gym.create_sphere(gym_instance.sim, sphere_dict['radius'], asset_options)
    sphere_dict['obs_handle'] = gym_instance.gym.create_actor(env_ptr, sphere_dict['obs_asset'], w_T_r * sphere_dict['pose'],
                                                 sphere_dict['name'], 2, 2, 0)
    sphere_dict['obs_body_handle'] = gym_instance.gym.get_actor_rigid_body_handle(env_ptr, sphere_dict['obs_handle'], 0)
    gym_instance.gym.set_rigid_body_color(env_ptr, sphere_dict['obs_handle'], 0, gymapi.MESH_VISUAL_AND_COLLISION,
                             gymapi.Vec3(0.8, 0.2, 0.2))
    return sphere_dict
