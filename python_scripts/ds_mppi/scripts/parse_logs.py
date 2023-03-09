import re
import numpy as np
np.set_printoptions(precision=3, suppress=True)

for i in [2, 3, 4, 5, 6, 7]:
    with open("../experiment_logs/my_%d.txt"%i, "r") as f:
        data = f.read().replace('\n', '')


    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", data)
    sz = len(numbers)
    numbers = [float(num) for num in numbers]

    dim = 8
    exp_arr = np.reshape(numbers, [int(sz/dim), dim])
    top_obs = exp_arr[:, 4:]

    # Filter out experiments where obstacle does not obstruct the motion significantly
    obs_mask = (top_obs[:, 0] < 0.8) & (top_obs[:, 2] > 0.6) & (top_obs[:, 0] > 0.2)

    # Filter out unsuccessful experiments
    success_mask = exp_arr[:, 0] == 1

    idx_obs = np.where(obs_mask)[0]
    idx_success = np.where(success_mask)[0]
    idx_obs_success = np.where(obs_mask & success_mask)[0]
    success = exp_arr[idx_obs, 0]
    n_iter = exp_arr[idx_obs_success, 1]
    time = exp_arr[idx_obs_success, 2]
    freq = exp_arr[idx_obs, 1] / exp_arr[idx_obs, 2]
    print('N: %d' % i,
          'Experiments: %d' % len(idx_obs),
          'Success: %4.2f' % success.mean(),
          'Time %4.2fs (%4.2f),' % (time.mean(), time.std()),
          'Iterations %4.2f (%4.2f),' % (n_iter.mean(), n_iter.std()),
          'Frequency: %4.2fHz (%4.2f)' % (freq.mean(), freq.std()))
