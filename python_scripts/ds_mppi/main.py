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


@torch.jit.script
def tangent_basis(normal_vec):
    A = torch.eye(normal_vec.shape[0]).to(normal_vec.device, normal_vec.dtype)
    A[:, 0] = normal_vec
    Q, R = torch.linalg.qr(A)
    norms = torch.linalg.norm(normal_vec)
    Q[:, 0] = normal_vec/torch.linalg.norm(normal_vec)
    return Q.nan_to_num()


@torch.jit.script
def tangent_basis_vec(normal_tens):
    T = torch.zeros((normal_tens.shape[0], normal_tens.shape[1], normal_tens.shape[1])).to(normal_tens.device, normal_tens.dtype)
    for i, n_v in enumerate(normal_tens):
        T[i] = tangent_basis(n_v)
    return T

def get_mindist(all_links, obs):
    mindists = dist_tens(all_links, obs)
    # calculate repulsions using fk_sym_gen
    idx_obs_closest = mindists[:, 1].to(torch.long).unsqueeze(1)
    idx_links_closest = mindists[:, 2].to(torch.long).unsqueeze(1)
    idx_pts_closest = mindists[:, 3].to(torch.long).unsqueeze(1)
    return mindists[:, 0], idx_obs_closest, idx_links_closest, idx_pts_closest

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
        # calculate nominal vector field
        nominal_velocity = (all_traj[:, i-1, :] - qf) @ A
        #nominal_velocity = nominal_velocity / torch.norm(nominal_velocity, dim=1).reshape(-1, 1)
        #all_traj[:, i, :] = all_traj[:, i-1, :] + dt * nominal_velocity
        # calculate modulations
        all_links, all_int_pts = numeric_fk_model_vec(all_traj[:, i-1, :], dh_params, 10)
        distance, idx_obs_closest, idx_links_closest, idx_pts_closest = get_mindist(all_links, obs)
        obs_pos_closest = obs[idx_obs_closest, 0:3].squeeze(1)
        int_points_closest = all_int_pts[torch.arange(N_traj).unsqueeze(1), idx_links_closest, idx_pts_closest].squeeze(1)
        rep_vec = lambda_rep_vec(all_traj[:, i-1, :], obs_pos_closest, idx_links_closest, int_points_closest, dh_a)
        rep_vec = rep_vec[:, 0:n_dof]
        E = tangent_basis_vec(rep_vec)
        # calculate standard modulation coefficients
        gamma = distance + 1
        gamma[gamma < 0] = 1e-8
        l_n = 1 - 1/gamma
        l_tau = 1 + 1/gamma
        l_n[l_n < 0] = 0
        l_tau[l_tau < 1] = 1
        D = (torch.zeros([N_traj, n_dof, n_dof])+torch.eye(n_dof)).to(q0.device, q0.dtype)
        D = l_tau[:, None, None] * D
        D[:, 0, 0] = l_n
        # build modulation matrix
        M = E @ D @ E.transpose(1, 2)
        # calculate modulated vector field (and normalize)
        mod_velocity = (M @ nominal_velocity.unsqueeze(2)).squeeze()
        mod_velocity_norm = torch.norm(mod_velocity, dim=1).reshape(-1, 1)
        mod_velocity_norm[mod_velocity_norm <= 1e-1] = 1
        mod_velocity = mod_velocity / mod_velocity_norm
        # slow down for collision case
        mod_velocity[distance < 0] *= 0.1
        # propagate
        all_traj[:, i, :] = all_traj[:, i-1, :] + dt * mod_velocity
    return all_traj

def main_int():
    # Initial state
    q_0 = torch.tensor([torch.pi/2, 0]).to(**params)
    q_f = torch.tensor([-torch.pi/2, 0]).to(**params)
    obs = torch.tensor([[5, 0, 0, 1]]).to(**params)
    n_dof = len(q_0)
    # Robot parameters
    dh_a = torch.tensor([0, 3, 3]).to(**params)
    dh_params = torch.vstack((dh_a*0, dh_a*0, dh_a, dh_a*0)).T
    # Integration parameters
    A = torch.diag(torch.tensor([-1, -1])).to(**params)
    N_traj = 10
    dt_H = 50
    dt = 0.2
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



