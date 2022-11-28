import numpy as np
from fk_num import *
from fk_sym import *
from plots import *
import numpy as np
import torch
import time

#define tensor parameters (cpu or cuda:0)
params = {'device': 'cpu', 'dtype': torch.float32}
#params = {'device': 'cuda:0', 'dtype': torch.float32}


def main():
    DOF = 2
    q = torch.zeros(DOF).to(**params)
    q[0] = np.pi/2
    dh_a = torch.tensor([0, 3, 3]).to(**params)
    dh_alpha = dh_a*0
    dh_d = dh_a*0
    dh_theta = dh_a*0
    dh_params = (torch.vstack((dh_d, dh_theta, dh_a, dh_alpha)).T).to(**params)
    link_pts, _ = numeric_fk_model(q, dh_params, 10)
    robot_sym = symbolic_fk_model(q, dh_params)
    r_h = init_robot_plot(link_pts, -10, 10, -10, 10)
    y_p = torch.tensor([8, 5]).to(**params)
    y_r = 1
    c_h = plot_circ(y_p, y_r)

    # for i in range(1000):
    #     q = q+torch.tensor([0.01, 0.01]).to(**params)
    #     link_pts = numeric_fk_model(q, dh_params, 10)['links']
    #     upd_r_h(link_pts, r_h)
    #     y_p = y_p-torch.tensor([0.01, 0.01]).to(**params)
    #     c_h.set_center(y_p)
    #     plt.pause(0.0001)

    t0 = time.time()
    for i in range(100000):
        link_pts, _ = numeric_fk_model(q, dh_params, 10)
    tf = time.time()
    print('numeric_fk_model time: ', tf-t0)
if __name__ == '__main__':
    main()



