import sys
sys.path.append('../ds_mppi/')
import numpy as np
from fk_num import *
from fk_sym_gen import *
import numpy as np
import torch
import time

params = {'device': 'cpu', 'dtype': torch.float32}
# robot DH parameters
dh_a = torch.tensor([0, 1, 1, 1, 1, 1])
dh_params = torch.vstack((dh_a * 0, dh_a * 0, dh_a, dh_a * 0)).T.to(**params)

DOF = len(dh_a)-1
# data ranges, joint limits, and point position limits
q_min = -np.pi*np.ones(DOF)*1.1
q_max = np.pi*np.ones(DOF)*1.1
p_min = np.array([-10, -10, 0])
p_max = np.array([10, 10, 0])

# number of samples
N_JPOS = 100
N_PPOS = 100
# generate random joint positions
rand_jpos = np.random.uniform(q_min, q_max, (N_JPOS, DOF))
rand_jpos = torch.tensor(rand_jpos).to(**params)
# compute forward kinematics for pregenerated joint positions
t0 = time.time()
all_fk, all_int_pts = numeric_fk_model_vec(rand_jpos, dh_params, 20)
# start main loop
data = torch.zeros(N_JPOS*N_PPOS, DOF+3+DOF) # joint position, point position, distances per link
for i in range(N_JPOS):
    print(i)
    rand_ppos = np.random.uniform(p_min, p_max, (N_PPOS, 3))
    rand_ppos = torch.tensor(rand_ppos).to(**params)
    for j in range(N_PPOS):
        dist = torch.norm(all_fk[i] - rand_ppos[j], 2, 2)
        res, _ = torch.min(dist, 1)
        data[i*N_PPOS+j] = torch.hstack((rand_jpos[i], rand_ppos[j], res))
print("time: %4.3f s"%(time.time()-t0))
torch.save(data, 'datasets/%d_dof_data_test.pt'%DOF)