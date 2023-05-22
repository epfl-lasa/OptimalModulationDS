import torch
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from matplotlib.colors import ListedColormap
import sys
sys.path.append('../functions/')
from plots import *
from fk_num import *
sys.path.append('../../mlp_learn/')
from sdf.robot_sdf import RobotSdfCollisionNet

params = {'device': 'cpu', 'dtype': torch.float32}

# robot parameters
DOF = 2
L = 3
dh_a = torch.zeros(DOF + 1)
dh_a[1:] = L  # link length
dh_params = torch.vstack((dh_a * 0, dh_a * 0, dh_a, dh_a * 0)).T

#plots and pointers
jpos_h = init_jpos_plot(-1.1*np.pi, 1.1*np.pi, -1.1*np.pi, 1.1*np.pi)
r_h = init_robot_plot(0, -10, 10, -10, 10)
obs = []
o_h = []
zero_contour = None

# controur
N_MESHGRID = 100
points_grid = torch.meshgrid([torch.linspace(-np.pi, np.pi, N_MESHGRID) for i in range(DOF)])
q_tens = torch.stack(points_grid, dim=-1).reshape(-1, DOF)

# nn loading
s = 256
n_layers = 5
skips = []
fname = '%ddof_sdf_%dx%d_mesh.pt' % (DOF, s, n_layers)
if skips == []:
    n_layers -= 1
nn_model = RobotSdfCollisionNet(in_channels=DOF + 3, out_channels=DOF, layers=[s] * n_layers, skips=skips)
nn_model.load_weights('../../../mlp_learn/models/' + fname, params)
nn_model.model.to(**params)
nn_model.model_jit = nn_model.model
nn_model.model_jit = torch.jit.script(nn_model.model_jit)
nn_model.model_jit = torch.jit.optimize_for_inference(nn_model.model_jit)
nn_model.update_aot_lambda()


def update_plot(*args):
    jspace = torch.tensor([slider1.get(), slider2.get()]).to(torch.float32)
    cur_fk, _ = numeric_fk_model(jspace, dh_params, 10)
    upd_r_h(cur_fk.to('cpu'), r_h)
    upd_jpos_plot(jspace, jpos_h)
    print('Current joint state: ', jspace)
def onclick(event):
    global obs, o_h
    if event.xdata > -10 and event.xdata < 10 and event.ydata > -10 and event.ydata < 10:
        # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       (event.button, event.x, event.y, event.xdata, event.ydata))
        removed = 0
        for h in o_h:
            h.remove()
        for i in range(len(obs)):
            if np.linalg.norm(np.array(obs[i][:2]) - np.array([event.xdata, event.ydata])) < obs[i][-1]:
                obs.pop(i)
                removed = 1
                break
        if removed == 0:
            obs.append([event.xdata, event.ydata, 0, 0.5])
        o_h = plot_obs_init(obs)
        update_contour_plot()
        print(obs)

def update_contour_plot(*args):
    global zero_contour
    obs_tens = torch.tensor(obs).to(**params)
    if obs_tens.shape[0] == 0:
        mindist_all = np.zeros((N_MESHGRID, N_MESHGRID))
    else:
        nn_input = torch.hstack((q_tens.tile(obs_tens.shape[0], 1), obs_tens.repeat_interleave(q_tens.shape[0], 0)))
        nn_dist = nn_model.model_jit(nn_input[:, 0:-1])
        nn_dist -= nn_input[:, -1].unsqueeze(1)
        mindist, _ = nn_dist.min(1)
        mindist_obst = mindist.reshape(-1, N_MESHGRID, N_MESHGRID).detach().numpy()
        mindist_all = mindist_obst.min(0)#.transpose(-1, 0)
    # print(mindist_all)
    carr = np.linspace([.1, .1, 1, 1], [1, 1, 1, 1], 256)
    fig = plt.figure(2)
    #clr_contoplt.contourf(points_grid[0], points_grid[1], mindist_all, levels=100, cmap=ListedColormap(carr))
    # contour zero lvl
    if zero_contour is not None:
        for c in zero_contour.collections:
            c.remove()
    zero_contour = plt.contour(points_grid[0], points_grid[1], mindist_all, levels=[0], colors='k')


cid = r_h.figure.canvas.mpl_connect('button_press_event', onclick)
root = tk.Tk()
root.title("Joint State")
root.geometry('+1900+700')
slider1 = tk.Scale(root, from_=-np.pi, to=np.pi, length=600,
                   resolution=0.01, tickinterval=np.pi/5, orient=tk.HORIZONTAL, command=update_plot)
slider1.grid(row=1, column=0)
slider2 = tk.Scale(root, from_=-np.pi, to=np.pi, length=600,
                   resolution=0.01, tickinterval=np.pi/5, orient=tk.HORIZONTAL, command=update_plot)
slider2.grid(row=2, column=0)
update_plot()
root.mainloop()
