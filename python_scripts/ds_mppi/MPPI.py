from policy import *
import sys

#@torch.jit.script
def tangent_basis(normal_vec):
    A = torch.eye(normal_vec.shape[0]).to(normal_vec.device, normal_vec.dtype)
    A[:, 0] = normal_vec
    Q, R = torch.linalg.qr(A)
    Q[:, 0] = normal_vec / torch.linalg.norm(normal_vec)
    return Q.nan_to_num()

#@torch.jit.script
def tangent_basis_vec(normal_tens, norm_basis):
    norm_basis = norm_basis * 0
    for i, n_v in enumerate(normal_tens):
        norm_basis[i] = tangent_basis(n_v)
    return norm_basis

def tangent_basis_tens(normal_tens):
    a = torch.eye(2).repeat(normal_tens.shape[0], 1).reshape(normal_tens.shape[0], normal_tens.shape[1], normal_tens.shape[1])
    a[:, :, 0] = normal_tens
    Q, R = torch.linalg.qr(a)
    return Q.nan_to_num()

class MPPI:
    def __init__(self, P, q0: torch.Tensor, qf: torch.Tensor, dh_params: torch.Tensor, obs: torch.Tensor, dt: float,
             dt_H: int, N_traj: int, A: torch.Tensor, dh_a: torch.Tensor, nn_model):
        self.tensor_args = {'device': q0.device, 'dtype': q0.dtype}
        self.P = P
        self.q0 = q0
        self.qf = qf
        self.dh_params = dh_params
        self.obs = obs
        self.dt = dt
        self.dt_H = dt_H
        self.N_traj = N_traj
        self.A = A
        self.dh_a = dh_a
        self.nn_model = nn_model
        self.n_dof = q0.shape[0]
        self.all_traj = torch.zeros(N_traj, dt_H, self.n_dof).to(**self.tensor_args)
        self.closest_dist_all = 100 + torch.zeros(N_traj, dt_H).to(**self.tensor_args)
        self.kernel_val_all = torch.zeros(N_traj, dt_H, P.N_KERNEL_MAX).to(**self.tensor_args)
        self.q_cur = q0
        self.nn_input = torch.zeros(N_traj * obs.shape[0], self.n_dof + 3).to(**self.tensor_args)
        self.D = (torch.zeros([self.N_traj, self.n_dof, self.n_dof]) + torch.eye(self.n_dof)).to(**self.tensor_args)
        self.nn_grad = torch.zeros(N_traj, self.n_dof).to(**self.tensor_args)
        self.norm_basis = torch.zeros((self.N_traj, self.n_dof, self.n_dof)).to(**self.tensor_args)
        self.basis_eye = torch.eye(2).repeat(N_traj, 1).reshape(N_traj, self.n_dof, self.n_dof)
        self.basis_eye_temp = self.basis_eye * 0
    def reset_tensors(self):
        self.all_traj = self.all_traj * 0
        self.closest_dist_all = 100 + self.closest_dist_all * 0
        self.kernel_val_all = self.kernel_val_all * 0

    def build_nn_input(self, q_tens, obs_tens):
        self.nn_input = torch.hstack((q_tens.tile(obs_tens.shape[0], 1), obs_tens.repeat_interleave(q_tens.shape[0], 0)))
        return self.nn_input

    def propagate(self):
        self.reset_tensors()
        self.all_traj[:, 0, :] = self.q_cur
        P = self.P
        for i in range(1, self.dt_H):
            with record_function("nominal vector field"):
                q_cur = self.all_traj[:, i, :]
                q_prev = self.all_traj[:, i - 1, :]
                # calculate nominal vector field
                nominal_velocity = (q_prev - self.qf) @ self.A
            # apply policies
            with record_function("apply policies"):
                kernel_value = eval_rbf(q_prev, P.mu_tmp[:, 0:P.n_kernels], P.sigma_tmp[:, 0:P.n_kernels])
                policy_value = torch.sum(P.alpha_tmp[:, 0:P.n_kernels] * kernel_value, 1)
                if P.n_kernels > 0:
                    self.kernel_val_all[:, i - 1, 0:P.n_kernels] = kernel_value.reshape((self.N_traj, P.n_kernels))
            with record_function("evaluate NN"):
                # evaluate NN
                nn_input = self.build_nn_input(q_prev, self.obs)
                nn_dist, nn_grad, nn_minidx = self.nn_model.compute_signed_distance_wgrad(nn_input[:, 0:-1], 'closest')
                nn_dist -= nn_input[:, -1].unsqueeze(1) # subtract radius
                nn_dist = nn_dist[torch.arange(self.N_traj).unsqueeze(1), nn_minidx.unsqueeze(1)]
                # get gradients
                self.nn_grad = nn_grad.squeeze(2)[:, 0:self.n_dof]
                distance = nn_dist.squeeze(1)
                self.closest_dist_all[:, i - 1] = distance
            with record_function("modulations"):
                # calculate modulations
                self.basis_eye_temp = self.basis_eye_temp*0 + self.basis_eye
                self.basis_eye_temp[:, :, 0] = self.nn_grad
                # QR decomposition
                E, R = torch.linalg.qr(self.basis_eye_temp)
                E[:, :, 0] = self.nn_grad / self.nn_grad.norm(2, 1).unsqueeze(1)
                # calculate standard modulation coefficients
                gamma = distance + 1 - 0.3
                gamma[gamma < 0] = 1e-8
                l_n = 1 - 1 / gamma
                l_tau = 1 + 1 / gamma
                l_n[l_n < 0] = 0
                l_tau[l_tau < 1] = 1
                # self.D = self.D * 0 + torch.eye(self.n_dof).to(**self.tensor_args)
                # self.D = l_tau[:, None, None] * self.D
                self.D = l_tau.repeat_interleave(self.n_dof).reshape((self.N_traj, 2)).diag_embed(0, 1, 2)
                self.D[:, 0, 0] = l_n
                # build modulation matrix
                M = E @ self.D @ E.transpose(1, 2)
            with record_function("apply policy"):
                # policy control
                policy_value[abs(policy_value) < 1e-6] = 0  # to normalize without errors
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
            with record_function("propagate"):
                # propagate
                self.all_traj[:, i, :] = self.all_traj[:, i - 1, :] + self.dt * mod_velocity
        return self.all_traj, self.closest_dist_all, self.kernel_val_all[:, :, 0:P.n_kernels]


