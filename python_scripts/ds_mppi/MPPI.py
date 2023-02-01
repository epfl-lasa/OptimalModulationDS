from policy import *
from cost import *
from fk_num import *
from fk_sym_gen import *

from torch.profiler import record_function


@torch.jit.script
def get_mindist(all_links, obs):
    mindists = dist_tens(all_links, obs)
    # calculate repulsions using fk_sym_gen
    idx_obs_closest = mindists[:, 1].to(torch.long).unsqueeze(1)
    idx_links_closest = mindists[:, 2].to(torch.long).unsqueeze(1)
    idx_pts_closest = mindists[:, 3].to(torch.long).unsqueeze(1)
    return mindists[:, 0], idx_obs_closest, idx_links_closest, idx_pts_closest

class MPPI:
    def __init__(self, q0: torch.Tensor, qf: torch.Tensor, dh_params: torch.Tensor, obs: torch.Tensor, dt: float,
             dt_H: int, N_traj: int, A: torch.Tensor, dh_a: torch.Tensor, nn_model):
        self.tensor_args = {'device': q0.device, 'dtype': q0.dtype}
        self.n_dof = q0.shape[0]
        self.Policy = TensorPolicyMPPI(N_traj, self.n_dof, self.tensor_args)
        self.q0 = q0
        self.qf = qf
        self.dh_params = dh_params
        self.obs = obs
        self.n_obs = obs.shape[0]
        self.dt = dt
        self.dt_H = dt_H
        self.N_traj = N_traj
        self.A = A
        self.dh_a = dh_a
        self.nn_model = nn_model
        self.all_traj = torch.zeros(N_traj, dt_H, self.n_dof).to(**self.tensor_args)
        self.closest_dist_all = 100 + torch.zeros(N_traj, dt_H).to(**self.tensor_args)
        self.kernel_val_all = torch.zeros(N_traj, dt_H, self.Policy.N_KERNEL_MAX).to(**self.tensor_args)
        self.q_cur = q0
        self.nn_input = torch.zeros(N_traj * obs.shape[0], self.n_dof + 3).to(**self.tensor_args)
        self.D = (torch.zeros([self.N_traj, self.n_dof, self.n_dof]) + torch.eye(self.n_dof)).to(**self.tensor_args)
        self.nn_grad = torch.zeros(N_traj, self.n_dof).to(**self.tensor_args)
        self.norm_basis = torch.zeros((self.N_traj, self.n_dof, self.n_dof)).to(**self.tensor_args)
        self.basis_eye = torch.eye(self.n_dof).repeat(N_traj, 1).reshape(N_traj, self.n_dof, self.n_dof).to(**self.tensor_args).cpu()
        self.basis_eye_temp = (self.basis_eye * 0).to(**self.tensor_args).cpu()
        self.nn_model.allocate_gradients(self.N_traj, self.tensor_args)
        self.Cost = Cost(self.qf, self.dh_params)
        self.traj_range = torch.arange(self.N_traj).to(**self.tensor_args).to(torch.long)
        self.policy_upd_rate = 0.1
        self.dst_thr = 0.5
        self.qdot = torch.zeros((self.N_traj, self.n_dof)).to(**self.tensor_args)
        self.ker_thr = 1e-3
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
        P = self.Policy
        for i in range(1, self.dt_H):
            with record_function("TAG: Nominal vector field"):
                q_prev = self.all_traj[:, i - 1, :]
                # calculate nominal vector field
                nominal_velocity = (q_prev - self.qf) @ self.A
            # apply policies
            with record_function("TAG: Apply policies"):
                kernel_value = eval_rbf(q_prev, P.mu_tmp[:, 0:P.n_kernels], P.sigma_tmp[:, 0:P.n_kernels])
                kernel_value[kernel_value < self.ker_thr] = 0
                ker_w = torch.exp(50*kernel_value)
                ker_w[kernel_value < self.ker_thr] = 0
                self.ker_w = torch.nan_to_num(ker_w / torch.sum(ker_w, 1).unsqueeze(1)) #normalize kernel influence
                policy_value = torch.sum(P.alpha_tmp[:, 0:P.n_kernels] * self.ker_w, 1)
                if P.n_kernels > 0:
                    self.kernel_val_all[:, i - 1, 0:P.n_kernels] = kernel_value.reshape((self.N_traj, P.n_kernels))
            #distance calculation (NN)
            with record_function("TAG: evaluate NN"):
                # evaluate NN
                distance, self.nn_grad = self.distance_repulsion_nn(q_prev)
                #distance, self.nn_grad = self.distance_repulsion_fk(q_prev) #not implemented for Franka
                distance -= self.dst_thr
                self.closest_dist_all[:, i - 1] = distance

            with record_function("TAG: QR decomposition"):
                # calculate modulations
                self.basis_eye_temp = self.basis_eye_temp*0 + self.basis_eye
                self.basis_eye_temp[:, :, 0] = self.nn_grad.cpu().to(torch.float32)
                E, R = torch.linalg.qr(self.basis_eye_temp)
                E = E.to(**self.tensor_args)
                E[:, :, 0] = self.nn_grad / self.nn_grad.norm(2, 1).unsqueeze(1)
            with record_function("TAG: Modulation-propagation"):
                # calculate standard modulation coefficients
                gamma = distance + 1
                gamma[gamma < 0] = 1e-8
                l_n = 1 - 1 / gamma
                l_tau = 1 + 1 / gamma
                l_n[l_n < 0] = 0
                l_tau[l_tau < 1] = 1
                # self.D = self.D * 0 + torch.eye(self.n_dof).to(**self.tensor_args)
                # self.D = l_tau[:, None, None] * self.D
                self.D = l_tau.repeat_interleave(self.n_dof).reshape((self.N_traj, self.n_dof)).diag_embed(0, 1, 2)
                self.D[:, 0, 0] = l_n
                # build modulation matrix
                M = E @ self.D @ E.transpose(1, 2)
            with record_function("TAG: Apply policy"):
                # policy control
                policy_value[abs(policy_value) < 1e-6] = 0  # to normalize without errors
                policy_value = torch.nan_to_num(policy_value / torch.norm(policy_value, 2, 1).unsqueeze(1))
                policy_velocity = (E[:, :, 1:] @ policy_value.unsqueeze(2)).squeeze(2)
                # calculate modulated vector field (and normalize)
                nominal_velocity_norm = torch.norm(nominal_velocity, dim=1).reshape(-1, 1)
                policy_velocity = policy_velocity * nominal_velocity_norm
                mod_velocity = (M @ (nominal_velocity + policy_velocity).unsqueeze(2)).squeeze()
                # normalization
                mod_velocity_norm = torch.norm(mod_velocity, dim=1).reshape(-1, 1)
                mod_velocity_norm[mod_velocity_norm <= 0.5] = 1
                mod_velocity = torch.nan_to_num(mod_velocity / mod_velocity_norm)
                # slow down for collision case
                mod_velocity[distance < 0] *= 0.1
            with record_function("TAG: Propagate"):
                # propagate
                self.all_traj[:, i, :] = self.all_traj[:, i - 1, :] + self.dt * mod_velocity
                if i == 1:
                    self.qdot = mod_velocity
        return self.all_traj, self.closest_dist_all, self.kernel_val_all[:, :, 0:P.n_kernels]

    def distance_repulsion_nn(self, q_prev):
        with record_function("TAG: evaluate NN_1 (build input)"):
            # building input tensor for NN (N_traj * n_obs, n_dof + 3)
            nn_input = self.build_nn_input(q_prev, self.obs)

        with record_function("TAG: evaluate NN_2 (forward pass)"):
            # doing single forward pass to figure out the closest obstacle for each configuration
            nn_dist = self.nn_model.model_jit.forward(nn_input[:, 0:-1])
            if self.nn_model.out_channels == 9:
                nn_dist = nn_dist/100   # scale down to meters
        with record_function("TAG: evaluate NN_3 (get closest obstacle)"):
            # rebuilding input tensor to only include closest obstacles
            nn_dist -= nn_input[:, -1].unsqueeze(1)  # subtract radius
            mindist, _ = nn_dist.min(1)
            mindist, sphere_idx = mindist.reshape(self.n_obs, self.N_traj).transpose(0, 1).min(1)
            mask_idx = self.traj_range + sphere_idx * self.N_traj
            nn_input = nn_input[mask_idx, :]

        with record_function("TAG: evaluate NN_4 (forward+backward pass)"):
            # forward + backward pass to get gradients for closest obstacles
            # nn_dist, nn_grad, nn_minidx = self.nn_model.compute_signed_distance_wgrad(nn_input[:, 0:-1], 'closest')
            # nn_dist, nn_grad, nn_minidx = self.nn_model.dist_grad_closest(nn_input[:, 0:-1])
            # self.nn_grad = nn_grad.squeeze(2)[:, 0:self.n_dof]

            nn_dist, nn_grad, nn_minidx = self.nn_model.dist_grad_closest_aot(nn_input[:, 0:-1])
            self.nn_grad = nn_grad[:, 0:self.n_dof]
            if self.nn_model.out_channels == 9:
                nn_dist = nn_dist/100   # scale down to meters

        with record_function("TAG: evaluate NN_5 (process outputs)"):
            # cleaning up to get distances and gradients for closest obstacles
            nn_dist -= nn_input[:, -1].unsqueeze(1)  # subtract radius and some threshold
            nn_dist = nn_dist[self.traj_range.unsqueeze(1), nn_minidx.unsqueeze(1)]
            # get gradients

            distance = nn_dist.squeeze(1)
        return distance, self.nn_grad

    def distance_repulsion_fk(self, q_prev):
        all_links, all_int_pts = numeric_fk_model_vec(q_prev, self.dh_params, 10)
        distance, idx_obs_closest, idx_links_closest, idx_pts_closest = get_mindist(all_links, self.obs)
        obs_pos_closest = self.obs[idx_obs_closest, 0:3].squeeze(1)
        int_points_closest = all_int_pts[self.traj_range.unsqueeze(1), idx_links_closest, idx_pts_closest].squeeze(1)
        rep_vec = lambda_rep_vec(q_prev, obs_pos_closest, idx_links_closest, int_points_closest, self.dh_a)
        self.nn_grad = rep_vec[:, 0:self.n_dof]
        return distance, rep_vec

    def get_cost(self):
        self.cur_cost = self.Cost.evaluate_costs(self.all_traj, self.closest_dist_all)
        return self.cur_cost

    def get_qdot(self, mode='best'):
        qdot = 0
        if mode == 'best':
            best_idx = torch.argmin(self.cur_cost)
            qdot = self.qdot[best_idx, :]
        elif mode == 'weighted':
            beta = self.cur_cost.mean() / 50
            w = torch.exp(-1 / beta * self.cur_cost)
            w = w / w.sum()
            qdot = torch.sum(w.unsqueeze(1) * self.qdot, dim=0)
        return qdot

    def shift_policy_means(self):
        beta = self.cur_cost.mean() / 50
        w = torch.exp(-1 / beta * self.cur_cost)
        w = w / w.sum()
        noupd_mask = self.kernel_val_all[:, :, 0:self.Policy.n_kernels].sum(1).mean(0) < self.ker_thr
        self.Policy.update_policy(w, self.policy_upd_rate, noupd_mask)
        return 0