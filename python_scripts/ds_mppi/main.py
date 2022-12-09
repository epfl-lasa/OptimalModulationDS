import numpy as np
from fk_num import *
from fk_sym_gen import *
from plots import *
import numpy as np
import torch
import time


#define tensor parameters (cpu or cuda:0)
if 1:
    params = {'device': 'cpu', 'dtype': torch.float32}
else:
    params = {'device': 'cuda:0', 'dtype': torch.float32}


def main():
    DOF = 2
    q_0 = torch.zeros(DOF).to(**params)
    q_0[0] = np.pi/2
    dh_a = torch.tensor([0, 3, 3]).to(**params)
    dh_params = torch.vstack((dh_a*0, dh_a*0, dh_a, dh_a*0)).T
    # link_pts, _ = numeric_fk_model(q, dh_params, 10)
    # robot_sym = symbolic_fk_model(q, dh_params)
    # r_h = init_robot_plot(link_pts, -10, 10, -10, 10)
    y_p = torch.tensor([0, 5, 0]).to(**params)
    y_r = 1
    # c_h = plot_circ(y_p, y_r)

    #t0 = time.time()
    #for i in range(100):
        #link_pts, v_links = numeric_fk_model(q, dh_params, 3)
        #dst = dist_to_point(link_pts, y_p)
        #print(dst)
        #p_closest = v_links[dst['linkidx']][dst['ptidx']].cpu().numpy()
        #mindist = robot_sym[dst['linkidx']]['dist'](q.cpu().numpy(), p_closest, y_p.cpu().numpy())
        #rep = robot_sym[dst['linkidx']]['rep'](q.cpu().numpy(), p_closest, y_p.cpu().numpy())
    #tf = time.time()
    #print('numeric_fk_model time: ', tf-t0)


#@torch.jit.script
def propagate_mod(q0: torch.Tensor,
                  qf: torch.Tensor,
                  dh_params: torch.Tensor,
                  obs: torch.Tensor,
                  dt: float,
                  dt_H: int,
                  N_traj: int,
                  A: torch.Tensor,
                  dh_a: torch.Tensor):
    n_dof = len(q0)
    all_traj = torch.zeros(N_traj, dt_H, n_dof).to(q0.device, q0.dtype)
    all_dists = torch.zeros(N_traj, obs.shape[0], 3).to(q0.device, q0.dtype)
    all_traj[:, 0, :] = q0
    for i in range(1, dt_H):
        all_vel = (all_traj[:, i-1, :] - qf) @ A
        all_vel = all_vel / torch.norm(all_vel, dim=1).reshape(-1, 1)
        all_traj[:, i, :] = all_traj[:, i-1, :] + dt * all_vel
        all_links, all_int_pts = numeric_fk_model_vec(all_traj[:, i, :], dh_params, 8)
        mindists = dist_tens(all_links, obs)
        #mindists = dist_to_points_vec(all_links, obs)
        # calculate repulsions using fk_sym_gen
        idx_obs_closest = mindists[:, 1].to(torch.long).unsqueeze(1)
        idx_links_closest = mindists[:, 2].to(torch.long).unsqueeze(1)
        idx_pts_closest = mindists[:, 3].to(torch.long).unsqueeze(1)
        obs_pos_closest = obs[idx_obs_closest, 0:3].squeeze(1)
        int_points_closest = all_int_pts[torch.arange(N_traj).unsqueeze(1), idx_links_closest, idx_pts_closest].squeeze(1)
        rep_vec = lambda_rep_vec(all_traj[:, i, :], obs_pos_closest, idx_links_closest, int_points_closest, dh_a)
    return all_traj

def main_int():
    # Initial state
    q_0 = torch.tensor([torch.pi/2, 0]).to(**params)
    q_f = torch.tensor([-torch.pi/2, 0]).to(**params)
    obs = torch.tensor([[10, 0, 0, 0.1], [5, 0, 0, 0.2], [2, 0, 0, 0.3]]).to(**params)
    n_dof = len(q_0)
    # Robot parameters
    dh_a = torch.tensor([0, 3, 3]).to(**params)
    dh_params = torch.vstack((dh_a*0, dh_a*0, dh_a, dh_a*0)).T
    # Integration parameters
    A = torch.diag(torch.tensor([-1, -1])).to(**params)
    N_traj = 20
    dt_H = 50
    dt = 0.01
    q_cur = q_0
    for ITER in range(50):
        if ITER == 3:
            t0 = time.time()
        dists = propagate_mod(q_cur, q_f, dh_params, obs, dt, dt_H, N_traj, A, dh_a)
    tf = time.time()
    td = tf-t0
    print('Time: ', td)
    print('Time per iteration: ', td/ITER, 'Hz: ', 1/(td/ITER))
    print('Time per rollout: ', td/(ITER*N_traj))
    print('Time per rollout step: ', td/(ITER*N_traj*dt_H))
    print('f')


if __name__ == '__main__':
    main_int()



