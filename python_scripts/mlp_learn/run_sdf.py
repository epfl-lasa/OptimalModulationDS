#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
from scipy.io import loadmat, savemat
import torch
import torch.nn.functional as F
import numpy as np
import time
import yaml

import os
import matplotlib.pyplot as plt
from sdf.robot_sdf import RobotSdfCollisionNet


def run_sdf():
    device = torch.device('cuda', 0)
    tensor_args = {'device': device, 'dtype': torch.float32}
    data = loadmat('../data-sampling/datasets/data_mesh_test.mat')['dataset']


    x = torch.Tensor(data[:, 0:10]).to(**tensor_args)
    y = 100 * torch.Tensor(data[:, 10:]).to(**tensor_args)

    dof = x.shape[1]
    s = 256
    n_layers = 5
    skips = []
    fname = 'sdf_%dx%d_mesh.pt'%(s,n_layers)
    if skips == []:
        n_layers-=1
    nn_model = RobotSdfCollisionNet(in_channels=dof, out_channels=y.shape[1], layers=[s] * n_layers, skips=skips)
    nn_model.load_weights('sdf_256x5_mesh.pt', tensor_args)

    nn_model.model.to(**tensor_args)

    model = nn_model.model
    nelem = sum([param.nelement() for param in model.parameters()])
    print(repr(model))
    print("Sum of parameters:%d" % nelem)

    # scale dataset: (disabled because of nerf features!)
    mean_x = torch.mean(x, dim=0) * 0.0
    std_x = torch.std(x, dim=0) * 0.0 + 1.0
    mean_y = torch.mean(y, dim=0) * 0.0
    std_y = torch.std(y, dim=0) * 0.0 + 1.0


    # print(x.shape)
    t0 = time.time()
    for i in range(1000):
        y_pred = model.forward(x)
    t1 = time.time()
    print("Time: %f" % (t1 - t0))
    y_pred = torch.mul(y_pred, std_y) + mean_y
    y_test = torch.mul(y, std_y) + mean_y
    # print(y_pred.shape, y_test.shape)
    loss = F.l1_loss(y_pred, y_test, reduction='mean')
    print(torch.median(y_pred), torch.mean(y_pred))
    print(loss.item())


if __name__ == '__main__':
    run_sdf()