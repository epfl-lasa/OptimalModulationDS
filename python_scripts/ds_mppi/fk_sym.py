import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from fk import *

def dh_transform_sym(q, d, theta, a, alpha):
    """
    Denavit-Hartenberg transformation matrix.
    """
    # Compute the transformation matrix
    sa = sp.sin(alpha)
    ca = sp.cos(alpha)
    sq = sp.sin(q+theta)
    cq = sp.cos(q+theta)
    T = sp.Matrix([
        [cq,        -sq,    0,      a],
        [sq * ca,   cq*ca,  -sa,    -d*sa],
        [sq * sa,   cq*sa,  ca,     d*ca],
        [0,         0,      0,      1]
    ])
    return T


def dh_fk_sym(q, dh_params):
    """
    Forward kinematics for a robot with Denavit-Hartenberg parameters.
    """
    # Initialize the transformation matrix
    T = [sp.eye(4)]
    # Loop through each joint
    for i, q_i in enumerate(q):
        # Extract the parameters for this joint
        d, theta, a, alpha = dh_params[i]
        # Compute the transformation for this joint
        T_prev = T[-1]
        T.append(T_prev @ dh_transform_sym(q_i, d, theta, a, alpha))
    return T


def symbolic_fk_model(q, dh_params):
    """
    Caclulate positions of points on the robot arm.
    """
    # Compute the transformation matrices
    n_dof = len(q)
    #q_sym = sp.symbols('q0:{}'.format(n_dof))
    q_sym = sp.MatrixSymbol('q', n_dof, 1)
    p_sym = sp.MatrixSymbol('p', 3, 1)
    P_arr = dh_fk_sym(q_sym, dh_params)
    all_links = []
    a = dh_params[:, 2]
    # Initialize the points array
    # Loop through each joint
    for i in range(n_dof):
        R = P_arr[i+1][:3, :3]
        T = P_arr[i+1][:3, 3]
        # Compute the position of the point for this link
        pos = R @ p_sym + T
        all_links.append(pos)
    return all_links


q = np.array([0, 0])
dh_a = np.array([0, 3, 3])
dh_alpha = dh_a*0
dh_d = dh_a*0
dh_theta = dh_a*0
dh_params = np.vstack((dh_d, dh_theta, dh_a, dh_alpha)).T
T_arr = dh_fk(q, dh_params)
robot = numeric_fk_model(q, dh_params, 10)
dst = dist_to_point(robot, np.array([10, 0, 0]))
robot_sym = symbolic_fk_model(q, dh_params)
l1 = robot_sym[1]
pos = sp.lambdify(['q', 'p'], l1, 'numpy')
y = robot['pts_int'][dst['linkidx']][dst['ptidx']]
print(pos(np.expand_dims(q, 1), np.expand_dims(y, 1)))
print('0')
#working symbolic position evaluation
#TODO: symbolic distance evaluation
#TODO: symbolic gradient evaluation



