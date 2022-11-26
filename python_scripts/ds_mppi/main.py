import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from fk import *


def main():
    q = np.array([0, 0])
    dh_a = np.array([0, 3, 3])
    dh_alpha = dh_a*0
    dh_d = dh_a*0
    dh_theta = dh_a*0
    dh_params = np.vstack((dh_d, dh_theta, dh_a, dh_alpha)).T
    T_arr = dh_fk(q, dh_params)
    link_pts = numeric_fk_model(q, dh_params, 10)
    r_h = init_robot_plot(link_pts, -10, 10, -10, 10)
    for i in range(1000):
        q = -np.array([i/10, i/10])
        link_pts = numeric_fk_model(q, dh_params, 10)
        upd_r_h(link_pts, r_h)
        plt.pause(0.0001)

if __name__ == '__main__':
    main()



