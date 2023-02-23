import torch
import time
import sys
from math import pi

sys.path.append('../functions/')
from fk_num import *
from plots import *

def rbf_kernel(q0, q, sigma, p=2):
    numerator = torch.norm(q - q0, p=p, dim=-1) ** 2
    exp_term = torch.exp(- sigma * numerator)
    return exp_term

#########################
### PARAMETERS SET UP ###
#########################
#sampling parameters
n_rand = 10000000
rand_scale = 1

# kernel parameters
p = 2
sigma = 0.1

rbf_thr_array =  [0.05, 0.1, 0.5, 0.9, 0.95]
n_plot = 100

clr_red = np.array([0.6, 0.05, 0.2])
clr_yellow = np.array([0.95, 0.7, 0.1])
clr_array = np.linspace(clr_yellow, clr_red, len(rbf_thr_array))

dh_a = torch.tensor([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0])  # "r" in matlab
dh_d = torch.tensor([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107])  # "d" in matlab
dh_alpha = torch.tensor([0, -pi / 2, pi / 2, pi / 2, -pi / 2, pi / 2, pi / 2, 0])  # "alpha" in matlab
dh_params = torch.vstack((dh_d, dh_a * 0, dh_a, dh_alpha)).T  # (d, theta, a (or r), alpha)
#dh_params = torch.vstack((torch.tensor([[0, 0, 0, 0]]), dh_params)) # add base link
#########################
###   execute script  ###
#########################

q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
q_max = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
delta = q_max - q_min
#init pose
q_0 = q_min + (q_max - q_min) * 0.5
fk_0 = numeric_fk_model(q_0, dh_params, 2)[0]

# init pose plot
h_0 = init_robot_plot3d(-1, 1, -1, 1, 0, 1, width=2, color=[0, 0.4470, 0.7410], markersize=2)
upd_r_h3d(fk_0, h_0)

#random poses batch
#q_rand = q_min + (q_max - q_min) * torch.rand(n_rand, 7) * rand_scale
q_rand = q_0 + (2*torch.rand(n_rand, 7)-1) * rand_scale

plotting = True
for p in [2]:
    for sigma in [0.5]:
        #rbf values
        rbf_val = rbf_kernel(q_0, q_rand, sigma, p)
        rbf_val_sorted, sorted_idx = rbf_val.sort()
        q_rand_sorted = q_rand[sorted_idx]
        arr_res = []
        for i, rbf_thr in enumerate(rbf_thr_array):
            idx_close = torch.where(rbf_val_sorted > rbf_thr)[0]
            rbf_val_close = rbf_val_sorted[idx_close]
            q_close_sorted = q_rand_sorted[idx_close]


            arr_res.append(100*idx_close.shape[0]/n_rand)

            # random poses plot
            if plotting:
                h_arr = []
                fk_arr = []
                for q in q_close_sorted[:n_plot]:
                    h_arr.append(init_line3d(width=1, color=clr_array[i], markersize=1))
                    fk_arr.append(numeric_fk_model(q, dh_params, 2)[0])
                    upd_r_h3d(fk_arr[-1], h_arr[-1])
                print(f'Number of poses close to q_0: {idx_close.shape[0]} [{100*idx_close.shape[0]/n_rand:4.2f}%] (within threshold of {rbf_thr})')
                print(rbf_val_sorted[n_plot]) if len(rbf_val_sorted) > n_plot else\
                    print(rbf_val_sorted[-1]) if len(rbf_val_sorted) > 0 else print('No poses found')
                h_0.set_zorder(10000)
        np.set_printoptions(precision=2, suppress=True)
        print(p, sigma)
        print(np.array2string(np.array(arr_res), separator=", "))
        if plotting:
            plt.pause(100)
