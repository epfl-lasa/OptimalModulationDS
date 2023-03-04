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
import torch
import torch.nn.functional as F
import numpy as np
import time
import yaml
from sdf_transformer import *

import os
import matplotlib.pyplot as plt
from sdf.robot_sdf import RobotSdfCollisionNet

device = torch.device('mps', 0)
tensor_args = {'device': device, 'dtype': torch.float32}
# q_dof = 7
# data = torch.load('datasets/%d_dof_data.pt' % q_dof).to(**tensor_args)
q_dof = 2
data = torch.load('datasets/2d_toy_data.pt').to(**tensor_args)

# data = data[0:1000000]
data_x = data[:, 0:q_dof + 2]
data_y = data[:, -q_dof:]
data_y = data[:, -1:] #2d toy

n_size = data.shape[0]
train_ratio = 0.98
test_ratio = 0.001
val_ratio = 1 - train_ratio - test_ratio

idx_train = np.arange(0, int(n_size * train_ratio))
idx_val = np.arange(idx_train[-1] + 1, int(n_size * (train_ratio + test_ratio)))
idx_test = np.arange(idx_val[-1] + 1, int(n_size))

x_train = data_x[idx_train, :]
y_train = data_y[idx_train, :]
x_val = data_x[idx_val, :]
y_val = data_y[idx_val, :]
x_test = data_x[idx_test, :]
y_test = data_y[idx_test, :]

# y[y<0]*=5
# y[y==0] = 1

s = 256
n_layers = 5
skips = []
fname = '%ddof_sdf_%dx%d_toy.pt' % (q_dof, s, n_layers)
if skips == []:
    n_layers -= 1
nn_model = RobotSdfCollisionNet(in_channels=x_train.shape[1], out_channels=y_train.shape[1], layers=[s] * n_layers,
                                skips=skips)
nn_model.load_weights('models/' + fname, tensor_args)

nn_model.model.to(**tensor_args)

model = nn_model.model


# model = sdf_transformer(input_dim=x_train.shape[1], output_dim=y_train.shape[1],
#                         num_layer=2, embed_dim=128, nhead=4, ff_dim=256)
# chk = torch.load('models/t_' + fname)
# model.load_state_dict(chk["model_state_dict"])

model.to(**tensor_args)
nelem = sum([param.nelement() for param in model.parameters()])
print(repr(model))
print("Sum of parameters:%d" % nelem)
# time.sleep(2)
# scale dataset: (disabled because of nerf features!)
mean_x = torch.mean(x_train, dim=0) * 0.0
std_x = torch.std(x_train, dim=0) * 0.0 + 1.0
mean_y = torch.mean(y_train, dim=0) * 0.0
std_y = torch.std(y_train, dim=0) * 0.0 + 1.0

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                       threshold=0.01, threshold_mode='rel',
                                                       cooldown=0, min_lr=0, eps=1e-04, verbose=True)
# print(model)
epochs = 100000
min_loss = 2000.0
# training:
e_notsaved = 0
scaler = torch.cuda.amp.GradScaler(enabled=True)
# train_midist, _ = torch.min(y_train, 1)
# val_midist, _ = torch.min(y_val, 1)
idx_close_train = (y_train[:, -1] < 1)
idx_close_val = (y_val[:, -1] < 1)

for e in range(epochs):
    t0 = time.time()
    model.train()
    loss = []
    i = 0
    idx_rand = torch.randperm(x_train.shape[0])[0:10000]
    idx_rand = torch.arange(0, x_train.shape[0]-1, 1)
    with torch.cuda.amp.autocast():
        y_pred_train = (model.forward(x_train[idx_rand, :]))
        train_loss = F.mse_loss(y_pred_train, y_train[idx_rand, :], reduction='mean')

    scaler.scale(train_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    loss.append(train_loss.item())

    model.eval()
    with torch.cuda.amp.autocast():
        y_pred = model.forward(x_val)
        val_loss = F.mse_loss(y_pred, y_val, reduction='mean')
        ee_loss_close = F.l1_loss(y_pred[idx_close_val][:, -1], y_val[idx_close_val][:, -1], reduction='mean')
    if e == 0:
        min_loss = val_loss
    scheduler.step(val_loss)
    train_loss = np.mean(loss)
    e_notsaved += 1
    if (val_loss < min_loss and e > 100 and e_notsaved > 100):
        e_notsaved = 0
        print('saving model', val_loss.item())
        torch.save(
            {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'norm': {'x': {'mean': mean_x, 'std': std_x},
                         'y': {'mean': mean_y, 'std': std_y}}
            },
            'models/' + fname)
        min_loss = val_loss
        print(y_pred[0, :])
        print(y_val[0, :])
        # if e > 1500:
        #     break
    print(
        "Epoch: %d (Saved at %d), Train Loss: %4.3f, Validation Loss: %4.3f (%4.3f), Epoch time: %4.3f s, LR = %4.8f" % (
            e, e - e_notsaved + 1, train_loss.item(), val_loss.item(), ee_loss_close.item(), time.time() - t0,
            optimizer.param_groups[0]["lr"]))
    # print(abs(y_pred-y_val).mean(0))

# with torch.no_grad():
#     x = x_test  # [y_test[:,0] > 0.0]
#     y = y_test  # [y_test[:,0] > 0.0]
#
#     # print(x.shape)
#     y_pred = model.forward(x)
#     y_pred = torch.mul(y_pred, std_y) + mean_y
#     y_test = torch.mul(y, std_y) + mean_y
#     print(y_test[y_test > 0.0])
#     print(y_pred[y_test > 0.0])
#     # print(y_pred.shape, y_test.shape)
#     loss = F.l1_loss(y_pred, y_test, reduction='mean')
#     print(torch.median(y_pred), torch.mean(y_pred))
#     print(loss.item())
