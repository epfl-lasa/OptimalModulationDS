from policy import *
from cost import *
from fk_num import *
from fk_sym_gen import *
from math import pi

from torch.profiler import record_function
import asyncio



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
        self.norm_basis = torch.zeros((self.N_traj, dt_H, self.n_dof, self.n_dof)).to(**self.tensor_args)
        self.basis_eye = torch.eye(self.n_dof).repeat(N_traj, 1).reshape(N_traj, self.n_dof, self.n_dof).to(**self.tensor_args).cpu()
        self.basis_eye_temp = (self.basis_eye * 0).to(**self.tensor_args).cpu()
        self.nn_model.allocate_gradients(self.N_traj+self.Policy.N_KERNEL_MAX, self.tensor_args)
        self.Cost = Cost(self.qf, self.dh_params)
        self.traj_range = torch.arange(self.N_traj).to(**self.tensor_args).to(torch.long)
        self.policy_upd_rate = 0.1
        self.dst_thr = 0.5
        self.qdot = torch.zeros((self.N_traj, self.n_dof)).to(**self.tensor_args)
        self.ker_thr = 1e-3

        self.kernel_gammas = torch.zeros(self.Policy.N_KERNEL_MAX, **self.tensor_args)
        self.kernel_obstacle_bases = torch.zeros((self.Policy.N_KERNEL_MAX, self.n_dof, self.n_dof), **self.tensor_args)

        for tmp in range(5):
            self.Policy.sample_policy()
            _, _, _ = self.propagate()
            numeric_fk_model(self.q_cur, dh_params, 10)
            print('Warmup done')

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
        if P.n_kernels > 0:
            self.update_kernel_normal_bases()
        for i in range(1, self.dt_H):
            with record_function("TAG: Nominal vector field"):
                q_prev = self.all_traj[:, i - 1, :]
                # calculate nominal vector field
                nominal_velocity = (q_prev - self.qf) @ self.A
            #distance calculation (NN)
            with record_function("TAG: evaluate NN"):
                # evaluate NN. Calculate kernel bases on first iteration
                distance, self.nn_grad = self.distance_repulsion_nn(q_prev, aot=True)

                #distance, self.nn_grad = self.distance_repulsion_nn(q_prev, aot=True)

                self.nn_grad = self.nn_grad[0:self.N_traj, :]               # fixes issue with aot_function cache
                # distance, self.nn_grad = self.distance_repulsion_fk(q_prev) #not implemented for Franka

                distance -= self.dst_thr
                self.closest_dist_all[:, i - 1] = distance[0:self.N_traj]
                distance = distance[0:self.N_traj]
            with record_function("TAG: QR decomposition"):
                # calculate modulations
                self.basis_eye_temp = self.basis_eye_temp*0 + self.basis_eye
                self.basis_eye_temp[:, :, 0] = self.nn_grad.cpu().to(torch.float32)
                E, R = torch.linalg.qr(self.basis_eye_temp)
                E = E.to(**self.tensor_args)
                E[:, :, 0] = self.nn_grad / self.nn_grad.norm(2, 1).unsqueeze(1)
                self.norm_basis[:, i-1] = E
            with record_function("TAG: Modulation-propagation"):
                # calculate standard modulation coefficients
                # gamma = distance + 1
                # gamma[gamma < 0] = 1e-8
                # l_n = 1 - 1 / gamma
                # l_tau = 1 + 1 / gamma
                # l_n[l_n < 0] = 0
                # l_tau[l_tau < 1] = 1
                # calculate own modulation coefficients
                if 0:
                    # for planar robot (units)
                    dist_low, dist_high = 0.5, 3
                    k_sigmoid = 3
                else:
                    # for franka robot (meters)
                    dist_low, dist_high = 0.01, 0.1
                    k_sigmoid = 100

                ln_min, ln_max = 0, 1
                ltau_min, ltau_max = 1, 10
                l_n = generalized_sigmoid(distance, ln_min, ln_max, dist_low, dist_high, k_sigmoid)
                l_tau = generalized_sigmoid(distance, ltau_max, ltau_min, dist_low, dist_high, k_sigmoid)
                # self.D = self.D * 0 + torch.eye(self.n_dof).to(**self.tensor_args)
                # self.D = l_tau[:, None, None] * self.D
                self.D = l_tau.repeat_interleave(self.n_dof).reshape((self.N_traj, self.n_dof)).diag_embed(0, 1, 2)
                self.D[:, 0, 0] = l_n
                # build modulation matrix
                M = E @ self.D @ E.transpose(1, 2)

            # apply policies
            with record_function("TAG: Apply policies"):
                tmp_basis = P.kernel_obstacle_bases[0:P.n_kernels]*0 + torch.eye(self.n_dof).to(**self.tensor_args)
                #ker_dist, ker_grad = self.distance_repulsion_nn(P.mu_c[0:P.n_kernels])
                kernel_value = eval_rbf(q_prev, P.mu_tmp[:, 0:P.n_kernels], P.sigma_tmp[:, 0:P.n_kernels], P.p)
                # kernel_value[kernel_value < self.ker_thr] = 0
                # ker_w = torch.exp(50*kernel_value)
                # ker_w[kernel_value < self.ker_thr] = 0
                # self.ker_w = torch.nan_to_num(ker_w / torch.sum(ker_w, 1).unsqueeze(1)) #normalize kernel influence
                self.ker_w = kernel_value
                # that's for kernel gamma(q_k) policy
                # P.alpha_tmp[:, 0:P.n_kernels, 0] = 0
                policy_all_flows = (P.kernel_obstacle_bases[0:P.n_kernels] @ P.alpha_tmp[:, 0:P.n_kernels].unsqueeze(3)).squeeze()
                # disable tangential stuff, optimize just some vector field
                # policy_all_flows = P.alpha_tmp[:, 0:P.n_kernels]
                # that's for local gamma(q) policy

                # workaround for broadcasting quirk TODO FIX
                if P.n_kernels == 1:
                    policy_value = (policy_all_flows * self.ker_w)[0]
                else:
                    policy_value = torch.sum(policy_all_flows * self.ker_w, 1)

                if P.n_kernels > 0:
                    self.kernel_val_all[:, i - 1, 0:P.n_kernels] = kernel_value.reshape((self.N_traj, P.n_kernels))

            with record_function("TAG: Apply policy"):
                # policy control
                ### policy depends on kernel Gamma(q_k) - assuming matrix multiplication done above
                #policy_velocity = policy_value
                policy_velocity = (1-l_n[:, None]) * policy_value

                #policy_velocity += ((1-l_n)/100).unsqueeze(1) * E[:, :, 0] #(some repuslion tweaking)
                # calculate modulated vector field (and normalize)
                nominal_velocity_norm = torch.norm(nominal_velocity, dim=1).reshape(-1, 1)
                # TODO CHECK POLICY VELOCITY NORMALIZATION (it's way less than 1)
                # policy_velocity = policy_velocity * nominal_velocity_norm             # magnitude of nominal velocity
                total_velocity = nominal_velocity + policy_velocity                   # magnitude of 2x nominal velocity
                # total_velocity_norm = torch.norm(total_velocity, dim=1).unsqueeze(1)
                # total_velocity_scaled = nominal_velocity_norm * total_velocity / total_velocity_norm
                mod_velocity = (M @ (total_velocity).unsqueeze(2)).squeeze()
                # # normalization
                mod_velocity_norm = torch.norm(mod_velocity, dim=-1).reshape(-1, 1)
                mod_velocity_norm[mod_velocity_norm <= 0.5] = 1
                mod_velocity = torch.nan_to_num(mod_velocity / mod_velocity_norm)
                # slow down and repulsion for collision case
                mod_velocity[distance < 0] *= 0.1
                repulsion_velocity = E[:, :, 0] * nominal_velocity_norm
                mod_velocity[distance < 0] += repulsion_velocity[distance < 0]
            with record_function("TAG: Propagate"):
                # propagate
                self.all_traj[:, i, :] = self.all_traj[:, i - 1, :] + self.dt * mod_velocity
                if i == 1:
                    self.qdot = mod_velocity
        return self.all_traj, self.closest_dist_all, self.kernel_val_all[:, :, 0:P.n_kernels]


    def distance_repulsion_nn(self, q_prev, aot=True):
        n_inputs = q_prev.shape[0]
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
            mindist, sphere_idx = mindist.reshape(self.n_obs, n_inputs).transpose(0, 1).min(1)
            #mask_idx = self.traj_range[:n_inputs] + sphere_idx * n_inputs
            mask_idx = torch.arange(n_inputs) + sphere_idx * n_inputs

            nn_input = nn_input[mask_idx, :]

        with record_function("TAG: evaluate NN_4 (forward+backward pass)"):
            # forward + backward pass to get gradients for closest obstacles
            # nn_dist, nn_grad, nn_minidx = self.nn_model.compute_signed_distance_wgrad(nn_input[:, 0:-1], 'closest')
            if aot:
                nn_dist, nn_grad, nn_minidx = self.nn_model.dist_grad_closest_aot(nn_input[:, 0:-1])
            else:
                nn_dist, nn_grad, nn_minidx = self.nn_model.dist_grad_closest(nn_input[:, 0:-1])
                nn_grad = nn_grad.squeeze(2)

            self.nn_grad = nn_grad[:, 0:self.n_dof]
            if self.nn_model.out_channels == 9:
                nn_dist = nn_dist/100   # scale down to meters

        with record_function("TAG: evaluate NN_5 (process outputs)"):
            # cleaning up to get distances and gradients for closest obstacles
            nn_dist -= nn_input[:, -1].unsqueeze(1)  # subtract radius and some threshold
            nn_dist = nn_dist[torch.arange(n_inputs).unsqueeze(1), nn_minidx.unsqueeze(1)]
            # get gradients

            distance = nn_dist.squeeze(1)
        return distance, self.nn_grad

    def update_kernel_normal_bases(self):
        # calculate gamma and repulsion for kernel centers
        dst, grad = self.distance_repulsion_nn(self.Policy.mu_c[0:self.Policy.n_kernels], aot=False)
        return 0
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
        max_kernel_activation_each = self.kernel_val_all[:, :, 0:self.Policy.n_kernels].max(dim=1)[0]
        mean_kernel_activation_all = max_kernel_activation_each.mean(dim=0)
        update_mask = mean_kernel_activation_all > self.ker_thr
        print(f'Updating {sum(update_mask)} kernels!')
        self.Policy.update_policy(w, self.policy_upd_rate, update_mask)
        return 0

    def update_obstacles(self, obs):
        self.obs = obs
        self.n_obs = obs.shape[0]
        return 0

def generalized_sigmoid(x, y_min, y_max, x0, x1, k):
    return y_min + (y_max - y_min) / (1 + torch.exp(k * (-x + (x0+x1)/2)))

