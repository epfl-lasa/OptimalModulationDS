import torch
from fk_num import *
from fk_sym_gen import *
from policy import *


@torch.jit.script
def tangent_basis(normal_vec):
    A = torch.eye(normal_vec.shape[0]).to(normal_vec.device, normal_vec.dtype)
    A[:, 0] = normal_vec
    Q, R = torch.linalg.qr(A)
    norms = torch.linalg.norm(normal_vec)
    Q[:, 0] = normal_vec / torch.linalg.norm(normal_vec)
    return Q.nan_to_num()


@torch.jit.script
def tangent_basis_vec(normal_tens):
    T = torch.zeros((normal_tens.shape[0], normal_tens.shape[1], normal_tens.shape[1])).to(normal_tens.device,
                                                                                           normal_tens.dtype)
    for i, n_v in enumerate(normal_tens):
        T[i] = tangent_basis(n_v)
    return T


@torch.jit.script
def get_mindist(all_links, obs):
    mindists = dist_tens(all_links, obs)
    # calculate repulsions using fk_sym_gen
    idx_obs_closest = mindists[:, 1].to(torch.long).unsqueeze(1)
    idx_links_closest = mindists[:, 2].to(torch.long).unsqueeze(1)
    idx_pts_closest = mindists[:, 3].to(torch.long).unsqueeze(1)
    return mindists[:, 0], idx_obs_closest, idx_links_closest, idx_pts_closest


# @torch.jit.script
def propagate_mod(q0: torch.Tensor,
                  qf: torch.Tensor,
                  dh_params: torch.Tensor,
                  obs: torch.Tensor,
                  dt: float,
                  dt_H: int,
                  N_traj: int,
                  A: torch.Tensor,
                  dh_a: torch.Tensor):
    n_dof = q0.shape[-1]
    all_traj = torch.zeros(N_traj, dt_H, n_dof).to(q0.device, q0.dtype)
    # all_dists = torch.zeros(N_traj, obs.shape[0], 3).to(q0.device, q0.dtype)
    all_traj[:, 0, :] = q0
    for i in range(1, dt_H):
        # calculate nominal vector field
        nominal_velocity = (all_traj[:, i - 1, :] - qf) @ A
        # calculate modulations
        all_links, all_int_pts = numeric_fk_model_vec(all_traj[:, i - 1, :], dh_params, 10)
        distance, idx_obs_closest, idx_links_closest, idx_pts_closest = get_mindist(all_links, obs)
        obs_pos_closest = obs[idx_obs_closest, 0:3].squeeze(1)
        int_points_closest = all_int_pts[torch.arange(N_traj).unsqueeze(1), idx_links_closest, idx_pts_closest].squeeze(
            1)
        rep_vec = lambda_rep_vec(all_traj[:, i - 1, :], obs_pos_closest, idx_links_closest, int_points_closest, dh_a)
        rep_vec = rep_vec[:, 0:n_dof]
        E = tangent_basis_vec(rep_vec)
        # calculate standard modulation coefficients
        gamma = distance + 1 - 0.1
        gamma[gamma < 0] = 1e-8
        l_n = 1 - 1 / gamma
        l_tau = 1 + 1 / gamma
        l_n[l_n < 0] = 0
        l_tau[l_tau < 1] = 1
        D = (torch.zeros([N_traj, n_dof, n_dof]) + torch.eye(n_dof)).to(all_traj.device, all_traj.dtype)
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
        all_traj[:, i, :] = all_traj[:, i - 1, :] + dt * mod_velocity
    return all_traj


def propagate_mod_policy(P,
                         q0: torch.Tensor,
                         qf: torch.Tensor,
                         dh_params: torch.Tensor,
                         obs: torch.Tensor,
                         dt: float,
                         dt_H: int,
                         N_traj: int,
                         A: torch.Tensor,
                         dh_a: torch.Tensor):
    n_dof = q0.shape[-1]
    all_traj = torch.zeros(N_traj, dt_H, n_dof).to(q0.device, q0.dtype)
    closests_dist_all = 100 + torch.zeros(N_traj, dt_H).to(q0.device, q0.dtype)
    kernel_val_all = torch.zeros(N_traj, dt_H, P.n_kernels).to(q0.device, q0.dtype)
    all_traj[:, 0, :] = q0
    for i in range(1, dt_H):
        # calculate nominal vector field
        nominal_velocity = (all_traj[:, i - 1, :] - qf) @ A
        # apply policies
        kernel_value = eval_rbf(all_traj[:, i - 1, :], P.mu_tmp[:, 0:P.n_kernels], P.sigma_tmp[:, 0:P.n_kernels])
        policy_value = torch.sum(P.alpha_tmp[:, 0:P.n_kernels] * kernel_value, 1)
        if P.n_kernels > 0:
            kernel_val_all[:, i - 1, :] = kernel_value.reshape((N_traj, P.n_kernels))
        # calculate modulations
        all_links, all_int_pts = numeric_fk_model_vec(all_traj[:, i - 1, :], dh_params, 10)
        distance, idx_obs_closest, idx_links_closest, idx_pts_closest = get_mindist(all_links, obs)
        closests_dist_all[:, i - 1] = distance
        obs_pos_closest = obs[idx_obs_closest, 0:3].squeeze(1)
        int_points_closest = all_int_pts[torch.arange(N_traj).unsqueeze(1), idx_links_closest, idx_pts_closest].squeeze(
            1)
        rep_vec = lambda_rep_vec(all_traj[:, i - 1, :], obs_pos_closest, idx_links_closest, int_points_closest, dh_a)
        rep_vec = rep_vec[:, 0:n_dof]
        E = tangent_basis_vec(rep_vec)
        # calculate standard modulation coefficients
        gamma = distance + 1 - 0.1
        gamma[gamma < 0] = 1e-8
        l_n = 1 - 1 / gamma
        l_tau = 1 + 1 / gamma
        l_n[l_n < 0] = 0
        l_tau[l_tau < 1] = 1
        D = (torch.zeros([N_traj, n_dof, n_dof]) + torch.eye(n_dof)).to(all_traj.device, all_traj.dtype)
        D = l_tau[:, None, None] * D
        D[:, 0, 0] = l_n
        # build modulation matrix
        M = E @ D @ E.transpose(1, 2)
        # policy control
        policy_value[policy_value < 1e-8] = 0
        policy_value = torch.nan_to_num(policy_value / torch.norm(policy_value, 2, 1).unsqueeze(1))
        policy_velocity = (E[:, :, 1:] @ policy_value.unsqueeze(2)).squeeze(2)
        # calculate modulated vector field (and normalize)
        nominal_velocity_norm = torch.norm(nominal_velocity, dim=1).reshape(-1, 1)
        policy_velocity = policy_velocity * nominal_velocity_norm
        mod_velocity = (M @ (nominal_velocity + policy_velocity).unsqueeze(2)).squeeze()
        mod_velocity_norm = torch.norm(mod_velocity, dim=1).reshape(-1, 1)
        mod_velocity_norm[mod_velocity_norm <= 1e-1] = 1
        mod_velocity = torch.nan_to_num(mod_velocity / mod_velocity_norm)
        # slow down for collision case
        mod_velocity[distance < 0] *= 0.1
        # propagate
        all_traj[:, i, :] = all_traj[:, i - 1, :] + dt * mod_velocity
    return all_traj, closests_dist_all, kernel_val_all


def propagate_mod_policy_nn(P,
                         q0: torch.Tensor,
                         qf: torch.Tensor,
                         dh_params: torch.Tensor,
                         obs: torch.Tensor,
                         dt: float,
                         dt_H: int,
                         N_traj: int,
                         A: torch.Tensor,
                         dh_a: torch.Tensor,
                         nn_model):
    n_dof = q0.shape[-1]
    all_traj = torch.zeros(N_traj, dt_H, n_dof).to(q0.device, q0.dtype)
    closests_dist_all = 100 + torch.zeros(N_traj, dt_H).to(q0.device, q0.dtype)
    kernel_val_all = torch.zeros(N_traj, dt_H, P.n_kernels).to(q0.device, q0.dtype)
    all_traj[:, 0, :] = q0
    for i in range(1, dt_H):
        # calculate nominal vector field
        nominal_velocity = (all_traj[:, i - 1, :] - qf) @ A
        # apply policies
        kernel_value = eval_rbf(all_traj[:, i - 1, :], P.mu_tmp[:, 0:P.n_kernels], P.sigma_tmp[:, 0:P.n_kernels])
        policy_value = torch.sum(P.alpha_tmp[:, 0:P.n_kernels] * kernel_value, 1)
        if P.n_kernels > 0:
            kernel_val_all[:, i - 1, :] = kernel_value.reshape((N_traj, P.n_kernels))
        # calculate modulations
        # evaluate NN
        nn_input = torch.hstack((all_traj[:, i - 1, :].tile(obs.shape[0], 1), obs.repeat_interleave(N_traj, 0)))
        nn_dist, nn_grad, nn_minidx = nn_model.compute_signed_distance_wgrad(nn_input[:, 0:-1], 'closest')
        # get mindistance
        nn_dist -= nn_input[:, -1].unsqueeze(1)
        nn_dist = nn_dist[torch.arange(N_traj).unsqueeze(1), nn_minidx.unsqueeze(1)]
        # get gradients
        nn_grad = nn_grad.squeeze(2)[:, 0:7]
        distance = nn_dist.squeeze(1)
        # all_links, all_int_pts = numeric_fk_model_vec(all_traj[:, i - 1, :], dh_params, 10)
        # distance, idx_obs_closest, idx_links_closest, idx_pts_closest = get_mindist(all_links, obs)
        # closests_dist_all[:, i - 1] = distance
        # obs_pos_closest = obs[idx_obs_closest, 0:3].squeeze(1)
        # int_points_closest = all_int_pts[torch.arange(N_traj).unsqueeze(1), idx_links_closest, idx_pts_closest].squeeze(1)
        # rep_vec = lambda_rep_vec(all_traj[:, i - 1, :], obs_pos_closest, idx_links_closest, int_points_closest, dh_a)
        # rep_vec = rep_vec[:, 0:n_dof]
        # E = tangent_basis_vec(rep_vec)
        E = tangent_basis_vec(nn_grad)

        # calculate standard modulation coefficients
        gamma = distance + 1 - 0.1
        gamma[gamma < 0] = 1e-8
        l_n = 1 - 1 / gamma
        l_tau = 1 + 1 / gamma
        l_n[l_n < 0] = 0
        l_tau[l_tau < 1] = 1
        D = (torch.zeros([N_traj, n_dof, n_dof]) + torch.eye(n_dof)).to(all_traj.device, all_traj.dtype)
        D = l_tau[:, None, None] * D
        D[:, 0, 0] = l_n
        # build modulation matrix
        M = E @ D @ E.transpose(1, 2)
        # policy control
        policy_value[policy_value < 1e-8] = 0
        policy_value = torch.nan_to_num(policy_value / torch.norm(policy_value, 2, 1).unsqueeze(1))
        policy_velocity = (E[:, :, 1:] @ policy_value.unsqueeze(2)).squeeze(2)
        # calculate modulated vector field (and normalize)
        nominal_velocity_norm = torch.norm(nominal_velocity, dim=1).reshape(-1, 1)
        policy_velocity = policy_velocity * nominal_velocity_norm
        mod_velocity = (M @ (nominal_velocity + policy_velocity).unsqueeze(2)).squeeze()
        mod_velocity_norm = torch.norm(mod_velocity, dim=1).reshape(-1, 1)
        mod_velocity_norm[mod_velocity_norm <= 1e-1] = 1
        mod_velocity = torch.nan_to_num(mod_velocity / mod_velocity_norm)
        # slow down for collision case
        mod_velocity[distance < 0] *= 0.1
        # propagate
        all_traj[:, i, :] = all_traj[:, i - 1, :] + dt * mod_velocity
    return all_traj, closests_dist_all, kernel_val_all
