import torch
import numpy as np
from sdf.robot_sdf import RobotSdfCollisionNet

tensor_args = {'device': 'cpu', 'dtype': torch.float32}


q_dof = 7
s = 256
n_layers = 5
skips = []
fname = '%ddof_sdf_%dx%d_mesh.pt' % (q_dof, s, n_layers)
if skips == []:
    n_layers -= 1
nn_model = RobotSdfCollisionNet(in_channels=q_dof+3, out_channels=q_dof, layers=[s] * n_layers, skips=skips)
nn_model.load_weights('models/' + fname, tensor_args)

model = nn_model.model
nelem = sum([param.nelement() for param in model.parameters()])

f = open("network2.txt", "a")
data = [len(model.layers[0])]
for i, layer in enumerate(model.layers[0]):
    W = layer[0].weight.detach().numpy()
    b = layer[0].bias.detach().numpy()
    shape = np.array(W.shape)
    print(i)
    np.savetxt(f, np.flip(shape).reshape(1, -1), fmt='%d')
    np.savetxt(f, W.transpose(), fmt='%f')
    np.savetxt(f, b.reshape(1, -1), fmt='%f')
f.close()
    # print(p.shape)
    # sz = np.array(p.shape)
    # weight = p.detach().numpy()
    # data.append(sz)
    # data.append(weight)
