import re
import numpy as np
np.set_printoptions(precision=3, suppress=True)

with open("../experiment_logs/my_3.txt", "r") as f:
    data = f.read().replace('\n', '')


numbers = re.findall(r"[-+]?\d*\.\d+|\d+", data)
sz = len(numbers)
numbers = [float(num) for num in numbers]

dim = 8
exp_arr = np.reshape(numbers, [int(sz/dim), dim])

success = exp_arr[:, 0]
n_iter = exp_arr[:, 1]
time = exp_arr[:, 2]
n_obs = exp_arr[:, 3]
top_obs = exp_arr[:, 4:]
freq = n_iter / time
print('Success: %4.2f' % success.mean(),
      'Time %4.2f s (%4.2f),' % (time.mean(), time.std()),
      'Iterations %4.2f s (%4.2f),' % (n_iter.mean(), n_iter.std()),
      'Frequency: %4.2f Hz (%4.2f)' % (freq.mean(), freq.std()))
