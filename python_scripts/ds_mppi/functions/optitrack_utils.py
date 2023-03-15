from scipy.spatial.transform import Rotation as ScipyRot
import numpy as np
import struct
import torch

def process_raw_message(data, tens_params):
    nBodies = int(len(data) / 36)
    bodies = []
    idx_goal = 0
    idx_obst = []
    for i in range(nBodies):
        subdata = data[i * 36:(i + 1) * 36]
        body_array = np.array(struct.unpack('iffffffff', subdata))
        tmp = dict()
        tmp['id'] = int(body_array[0])
        tmp['pos'] = torch.tensor(body_array[2:5] - np.array([0, 0.2, 0])).to(**tens_params)
        tmp['rot_tens'] = torch.tensor([body_array[6], body_array[7], body_array[8], body_array[5]]).to(**tens_params)
        tmp['rot'] = torch.tensor(ScipyRot.from_quat(tmp['rot_tens'].numpy()).as_matrix()).to(**tens_params)
        bodies.append(tmp)

    return(bodies)