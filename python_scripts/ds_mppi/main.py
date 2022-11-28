import numpy as np
from fk_num import *
from fk_sym import *
from plots import *
import numpy as np
import torch
import time

#define tensor parameters (cpu or cuda:0)
if 1:
    params = {'device': 'cpu', 'dtype': torch.float32}
else:
    params = {'device': 'cuda:0', 'dtype': torch.float32}

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
    # r_h = init_robot_plot(link_pts, -10, 10, -10, 10)
    y_p = torch.tensor([8, 5, 0]).to(**params)
    y_r = 1
    # c_h = plot_circ(y_p, y_r)

    #warmup jit script
    torch.jit.optimized_execution(True)
    link_pts, v_links = numeric_fk_model(q, dh_params, 10)
    dst = dist_to_point(link_pts, y_p)

    t0 = time.time()
    for i in range(1000):
        link_pts, v_links = numeric_fk_model(q, dh_params, 3)
        dst = dist_to_point(link_pts, y_p)
        #print(dst)
        p_closest = v_links[dst['linkidx']][dst['ptidx']].cpu().numpy()
        #mindist = robot_sym[dst['linkidx']]['dist'](q.cpu().numpy(), p_closest, y_p.cpu().numpy())
        #rep = robot_sym[dst['linkidx']]['rep'](q.cpu().numpy(), p_closest, y_p.cpu().numpy())
    tf = time.time()
    print('numeric_fk_model time: ', tf-t0)
if __name__ == '__main__':
    main()



