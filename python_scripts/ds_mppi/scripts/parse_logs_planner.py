import re
import numpy as np
import os

np.set_printoptions(precision=3, suppress=True)

files_list = []
base_path = "../experiment_logs/planner_logs/5/"
for file in os.listdir(base_path):
    if file.endswith(".txt"):
        files_list.append(base_path+file)

print(files_list)
mpc_act_rates = []
iters = []
for fname in files_list:
    print('Parsing file %s'%fname)
    exp_data = np.loadtxt(fname)
    print(exp_data)
    n_iter = exp_data.shape[1]
    iters.append(n_iter)
    n_noupd = np.sum(exp_data[0] == 0)
    mpc_act_rates.append(1 - n_noupd/n_iter)
    print('Number of iterations: %d' % n_iter)
    print('Number of iterations without kernel update: %d' % n_noupd)
    print('MPC activation rate: %f' % (1 - n_noupd/n_iter))

idx_nompc = np.array(mpc_act_rates) == 0
arr_mpc_act = np.array(mpc_act_rates)[~idx_nompc]
print('Average iteration number: %f' % np.mean(iters))
print('Std iteration number: %f' % np.std(iters))
print('Trajectories without MPC: %d, Rate: %f' % (np.sum(idx_nompc), np.sum(idx_nompc)/len(mpc_act_rates)))
print('Average MPC activation rate: %f' % np.mean(arr_mpc_act))
print('Std MPC activation rate: %f' % np.std(arr_mpc_act))
# for i in [2, 3, 4, 5, 6, 7]:
#     with open("../experiment_logs/storm_%d.txt"%i, "r") as f:
#         data = f.read().replace('\n', '')
#
#
#     numbers = re.findall(r"[-+]?\d*\.\d+|\d+", data)
#     sz = len(numbers)
#     numbers = [float(num) for num in numbers]
#
#     dim = 8
#     exp_arr = np.reshape(numbers, [int(sz/dim), dim])
#     top_obs = exp_arr[:, 4:]
#
#     # Filter out experiments where obstacle does not obstruct the motion significantly
#     obs_mask = (top_obs[:, 0] < 0.8) & (top_obs[:, 2] > 0.6) & (top_obs[:, 0] > 0.2)
#
#     # Filter out unsuccessful experiments
#     success_mask = exp_arr[:, 0] == 1
#
#     idx_obs = np.where(obs_mask)[0]
#     idx_success = np.where(success_mask)[0]
#     idx_obs_success = np.where(obs_mask & success_mask)[0]
#     success = exp_arr[idx_obs, 0]
#     n_iter = exp_arr[idx_obs_success, 1]
#     time = exp_arr[idx_obs_success, 2]
#     freq = exp_arr[idx_obs, 1] / exp_arr[idx_obs, 2]
#     print('N: %d' % i,
#           'Experiments: %d' % len(idx_obs),
#           'Success: %4.2f' % success.mean(),
#           'Time %4.2fs (%4.2f),' % (time.mean(), time.std()),
#           'Iterations %4.2f (%4.2f),' % (n_iter.mean(), n_iter.std()),
#           'Frequency: %4.2fHz (%4.2f)' % (freq.mean(), freq.std()))
