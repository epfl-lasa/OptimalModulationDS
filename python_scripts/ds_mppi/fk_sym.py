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
    q_sym = sp.Matrix(sp.MatrixSymbol('q', n_dof, 1))
    p_sym = sp.Matrix(sp.MatrixSymbol('p', 3, 1))
    y_sym = sp.Matrix(sp.MatrixSymbol('y', 3, 1))
    P_arr = dh_fk_sym(q_sym, dh_params)
    all_links = []
    a = dh_params[:, 2]
    # Initialize the points array
    # Loop through each joint
    for i in range(n_dof):
        link_dict = dict()
        R = P_arr[i+1][:3, :3]
        T = P_arr[i+1][:3, 3]
        # Compute the position of the point on this link
        pos = R @ p_sym + T
        #distance for point on link to task space point
        dist = sp.sqrt(((pos - y_sym).T * (pos - y_sym))[0])
        #task space gradient
        ddist = 1/dist * (pos - y_sym)
        #jacobian
        J = pos.jacobian(q_sym)
        #repulsion
        rep = (ddist.T * J)
        pos_f = sp.lambdify(['q', 'p'], pos, 'numpy')
        dst_f = sp.lambdify(['q', 'p', 'y'], dist, 'numpy')
        rep_f = sp.lambdify(['q', 'p', 'y'], rep, 'numpy')

        link_dict['pos'] = lambda q, p: pos_f(np.expand_dims(q, 1), np.expand_dims(p, 1)).reshape(-1)
        link_dict['dist'] = lambda q, p, y: dst_f(np.expand_dims(q, 1), np.expand_dims(p, 1), np.expand_dims(y, 1))
        link_dict['rep'] = lambda q, p, y: rep_f(np.expand_dims(q, 1), np.expand_dims(p, 1), np.expand_dims(y, 1)).reshape(-1)

        all_links.append(link_dict)
    return all_links

def main():
    q = np.array([0, 0, 0])
    dh_a = np.array([0, 3, 3, 3])
    dh_alpha = dh_a * 0
    dh_d = dh_a * 0
    dh_theta = dh_a * 0
    dh_params = np.vstack((dh_d, dh_theta, dh_a, dh_alpha)).T
    robot = numeric_fk_model(q, dh_params, 10)
    y = np.array([10, 1, 0])
    dst = dist_to_point(robot, y)
    robot_sym = symbolic_fk_model(q, dh_params)
    l1 = robot_sym[1]

    p = robot['pts_int'][dst['linkidx']][dst['ptidx']]
    print(l1['pos'](q, p))
    print(l1['dist'](q, p, y))
    print(l1['rep'](q, p, y))

    print('fin')


if __name__ == '__main__':
    main()



