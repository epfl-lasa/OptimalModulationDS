import numpy as np
from fk_num import *
from fk_sym import *
from plots import *

def main():
    DOF = 2
    q = np.zeros(DOF)
    q[0] = np.pi/2
    dh_a = np.array([0, 3, 3])
    dh_alpha = dh_a*0
    dh_d = dh_a*0
    dh_theta = dh_a*0
    dh_params = np.vstack((dh_d, dh_theta, dh_a, dh_alpha)).T
    robot_num = numeric_fk_model(q, dh_params, 10)
    robot_sym = symbolic_fk_model(q, dh_params)
    r_h = init_robot_plot(robot_num['links'], -10, 10, -10, 10)
    y_p = np.array([8, 5])
    y_r = 1
    c_h = plot_circ(y_p, y_r)

    for i in range(1000):
        q = q+np.array([0.01, 0.01])
        link_pts = numeric_fk_model(q, dh_params, 10)['links']
        upd_r_h(link_pts, r_h)
        y_p = y_p-np.array([0.01, 0.01])
        c_h.set_center(y_p)
        plt.pause(0.0001)

if __name__ == '__main__':
    main()



