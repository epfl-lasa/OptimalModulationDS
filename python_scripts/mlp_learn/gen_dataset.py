import sys
sys.path.append('../ds_mppi/')
import numpy as np
from fk_num import *
from fk_sym_gen import *
import numpy as np
import torch
import time

params = {'device': 'cpu', 'dtype': torch.float32}
DOF = 2
q_min = np.array([-np.pi, -np.pi])
q_max = np.array([np.pi, np.pi])
p_min = np.array([-10, -10, 0, 0])
p_max = np.array([10, 10, 0, 0])

all_min = np.concatenate((q_min, p_min))
all_max = np.concatenate((q_max, p_max))

dh_a = torch.tensor([0, 3, 3])
dh_params = torch.vstack((dh_a * 0, dh_a * 0, dh_a, dh_a * 0)).T.to(**params)

N_JPOS = 1000
N_PPOS = 1000
rand_jpos = np.random.uniform(q_min, q_max, (N_JPOS, DOF))
rand_jpos = torch.tensor(rand_jpos).to(**params)
t0 = time.time()
all_fk, all_int_pts = numeric_fk_model_vec(rand_jpos, dh_params, 20)

data = torch.zeros(N_JPOS*N_PPOS, DOF+3+1)
for i in range(N_JPOS):
    print(i)
    rand_ppos = np.random.uniform(p_min, p_max, (N_PPOS, 4))
    rand_ppos = torch.tensor(rand_ppos).to(**params)
    for j in range(N_PPOS):
        res = dist_to_point(all_fk[i], rand_ppos[j])
        data[i*N_PPOS+j] = torch.hstack((res[0], rand_jpos[i], rand_ppos[j, 0:3]))
print("time: %4.3f s"%(time.time()-t0))
torch.save(data, 'data.pt')