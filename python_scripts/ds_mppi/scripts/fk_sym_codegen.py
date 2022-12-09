import matplotlib.pyplot as plt
import sympy as sp
import numpy as np


def dh_transform_sym(q, d, theta, a, alpha):
    """
    Denavit-Hartenberg (modified) transformation matrix.
    """
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
        d, theta, a, alpha = dh_params[i, :]
        theta = 0
        d = 0
        alpha = 0
        # Compute the transformation for this joint
        T_prev = T[-1]
        T.append(T_prev @ dh_transform_sym(q_i, d, theta, a, alpha))
    return T


def symbolic_fk_model(n_dof):
    """
    Build symbolic model of the robot
    Provide lambda functions for position, distance, and joint repulsion
    """
    # Compute the transformation matrices
    q_sym = sp.Matrix(sp.MatrixSymbol('q', n_dof, 1))
    p_sym = sp.Matrix(sp.MatrixSymbol('p', 3, 1))
    y_sym = sp.Matrix(sp.MatrixSymbol('y', 3, 1))
    alphas = sp.symbols('alpha0:{}'.format(n_dof))
    a = sp.symbols('a0:{}'.format(n_dof))
    d = sp.symbols('d0:{}'.format(n_dof))
    theta = sp.symbols('theta0:{}'.format(n_dof))
    dh_params = sp.Matrix([d, theta, a, alphas]).T
    P_arr = dh_fk_sym(q_sym, dh_params)

    all_links = []
    a = dh_params[:, 2]
    # Initialize the points array
    # Loop through each joint
    for i in range(n_dof):
        print(i)
        link_dict = dict()
        R = P_arr[i+1][:3, :3]
        T = P_arr[i+1][:3, 3]
        # Compute the position of the point on this link
        pos = R @ p_sym + T
        # Distance for point on link to task space point
        dist = sp.sqrt(((pos - y_sym).T * (pos - y_sym))[0])
        # Task space gradient
        ddist = 1/dist * (pos - y_sym)
        # Jacobian
        J = pos.jacobian(q_sym)
        # Repulsion
        rep = ddist.T * J
        link_dict['pos'] = pos
        link_dict['dist'] = dist
        link_dict['rep'] = rep

        all_links.append(link_dict)
    return all_links

def main():
    # Define the number of DOF
    n_dof = 7
    # Build the symbolic model
    all_links = symbolic_fk_model(n_dof)
    # Print the symbolic expressions for the distance and repulsion
    with open('sym.txt', 'w') as f:
        for i, link in enumerate(all_links):
            f.writelines(str(f'\nLink {i+1}'))
            f.writelines(str('\nPosition:\n'))
            f.writelines(str(sp.simplify(link['pos'])))
            f.writelines(str('\nDistance\n'))
            f.writelines(str(sp.simplify(link['dist'])))
            f.writelines(str('\nRepulsion\n'))
            f.writelines(str(sp.simplify(link['rep'])))

if __name__ == '__main__':
    main()



