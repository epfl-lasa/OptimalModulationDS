import torch
from fk_num import *
from fk_sym_gen import *
import time
from torch.profiler import profile, record_function, ProfilerActivity


# Class defining MPPI policy for placing RBF kernels in the state space
# This policy is defined as a set of RBF kernels (placed in the N_DOF-dimensional state space of the robot)
# Each RBF kernel has fixed width and is centered at a point in the state space where sampling-based exploration encountered an obstacle
# Kernels are applied to modulated DS to add perturbations along tangent directions (N_DOF-1 dimensional space)
# Kernels are sampled from MPPI policy (so kernel centers, widths and weights are sampled from a Gaussian distribution)
class TensorPolicyMPPI:
    def __init__(self, n_traj, n_dof, tensor_params):
        self.n_dof = n_dof
        self.n_traj = n_traj
        self.params = tensor_params
        self.n_kernels = 0
        self.N_KERNEL_MAX = 1000  # Maximum number of kernels
        self.sigma_c_nominal = 0.2  # Nominal standard deviation of RBF kernels
        # Policy centers
        self.mu_c = torch.zeros((self.N_KERNEL_MAX, self.n_dof), **self.params)  # Centers of RBF kernels
        self.sigma_c = torch.zeros((self.N_KERNEL_MAX), **self.params)  # Standard deviations of RBF kernels
        self.alpha_c = torch.zeros((self.N_KERNEL_MAX, self.n_dof - 1),
                                   **self.params)  # Weights of RBF kernels for each tangential direction
        # Policy sigmas
        self.mu_s = torch.tensor(0, **self.params)
        self.sigma_s = torch.tensor(0.0, **self.params)
        self.alpha_s = torch.tensor(0.5, **self.params)
        # sampled policy
        self.mu_tmp = torch.zeros((self.n_traj, self.N_KERNEL_MAX, self.n_dof), **self.params)
        self.sigma_tmp = torch.zeros((self.n_traj, self.N_KERNEL_MAX), **self.params)
        self.alpha_tmp = torch.zeros((self.n_traj, self.N_KERNEL_MAX, self.n_dof - 1), **self.params)

    def sample_policy(self):
        # Sample centers
        # self.mu_tmp = torch.randn((self.n_traj, self.N_KERNEL_MAX, self.n_dof), **self.params) * torch.sqrt(self.mu_s) + self.mu_c
        # self.sigma_tmp = torch.randn((self.n_traj, self.N_KERNEL_MAX), **self.params) * torch.sqrt(self.sigma_s) + self.sigma_c
        # self.alpha_tmp = torch.randn((self.n_traj, self.N_KERNEL_MAX, self.n_dof-1), **self.params) * torch.sqrt(self.alpha_s) + self.alpha_c
        # previous implementation (copying from CPU!)
        # self.mu_tmp[:, :self.n_kernels] = torch.randn((self.n_traj, self.n_kernels, self.n_dof),
        #                                               **self.params) * self.mu_s + self.mu_c[:self.n_kernels]
        # self.sigma_tmp[:, :self.n_kernels] = torch.randn((self.n_traj, self.n_kernels),
        #                                                  **self.params) * self.sigma_s + self.sigma_c[:self.n_kernels]
        # self.alpha_tmp[:, :self.n_kernels] = torch.randn((self.n_traj, self.n_kernels, self.n_dof - 1),
        #                                                  **self.params) * self.alpha_s + self.alpha_c[:self.n_kernels]
        # GPU sampling
        self.mu_tmp[:, :self.n_kernels] = self.mu_tmp[:, :self.n_kernels].normal_(mean=0, std=self.mu_s) \
                                          + self.mu_c[:self.n_kernels]
        self.sigma_tmp[:, :self.n_kernels] = self.sigma_tmp[:, :self.n_kernels].normal_(mean=0, std=self.sigma_s) \
                                                + self.sigma_c[:self.n_kernels]
        self.alpha_tmp[:, :self.n_kernels] = self.alpha_tmp[:, :self.n_kernels].normal_(mean=0, std=self.alpha_s) \
                                                + self.alpha_c[:self.n_kernels]
    def update_policy(self, w, upd_rate, noupd_mask=None):
        # Update policy parameters, use current state(mu_c, sigma_c and alpha_c)
        # and a sampled policy (mu_tmp, sigma_tmp, alpha_tmp) with weights w
        mu_cur_sum = torch.sum(w[:, None, None] * self.mu_tmp[:, :self.n_kernels], 0)
        sigma_cur_sum = torch.sum(w[:, None] * self.sigma_tmp[:, :self.n_kernels], 0)
        alpha_cur_sum = torch.sum(w[:, None, None] * self.alpha_tmp[:, :self.n_kernels], 0)

        if self.n_kernels > 0:
            # prepare upd rate tensor
            upd_tens = upd_rate*torch.ones(self.n_kernels, **self.params)
            if noupd_mask is not None:
                upd_tens[noupd_mask] = 0.0
            # Update centers
            # scalar upd_rate
            # self.mu_c[0:self.n_kernels] = (1 - upd_rate) * self.mu_c[0:self.n_kernels] + upd_rate * mu_cur_sum
            # # Update standard deviations
            # self.sigma_c[0:self.n_kernels] = (1 - upd_rate) * self.sigma_c[0:self.n_kernels] + upd_rate * sigma_cur_sum
            # # Update weights
            # self.alpha_c[0:self.n_kernels] = (1 - upd_rate) * self.alpha_c[0:self.n_kernels] + upd_rate * alpha_cur_sum

            #tensor upd_rate
            self.mu_c[0:self.n_kernels] = (1 - upd_tens[:, None]) * self.mu_c[0:self.n_kernels] + upd_tens[:, None] * mu_cur_sum
            # Update standard deviations
            self.sigma_c[0:self.n_kernels] = (1 - upd_tens) * self.sigma_c[0:self.n_kernels] + upd_tens * sigma_cur_sum
            # Update weights
            self.alpha_c[0:self.n_kernels] = (1 - upd_tens[:, None]) * self.alpha_c[0:self.n_kernels] + upd_tens[:, None] * alpha_cur_sum

    def add_kernel(self, q):
        # Add a new kernel centered at q
        if self.n_kernels < self.N_KERNEL_MAX:
            # new center as specified
            self.mu_c[self.n_kernels, :] = q
            # new standard deviation as nominal
            self.sigma_c[self.n_kernels] = self.sigma_c_nominal
            # new weights same as weights of the closest kernel
            if self.n_kernels > 0:
                _, idx = torch.min(torch.norm(self.mu_c[0:self.n_kernels, :] - q, 2, 1), 0)
                self.alpha_c[self.n_kernels, :] = self.alpha_c[idx, :]
            else:
                self.alpha_c[self.n_kernels, :] = 0
            # increment number of kernels
            self.n_kernels += 1
        else:
            print('Not adding new kernel at: maximum number of kernels reached', q)

    def check_traj_for_kernels(self, all_traj, closests_dist_all, thr_dist, thr_kernel):
        # Check if a trajectory has encountered an obstacle
        # If yes, output this position as potential kernel center
        # all_traj: all trajectories
        # closests_dist_all: distance to closest obstacle for each point of each trajectory
        # kernel_val_all: value of RBF kernels for each point of each trajectory
        # return: potential kernel centers
        idx_close = closests_dist_all < thr_dist
        dist_ker_centers = all_traj - self.mu_c[0:self.n_kernels].unsqueeze(0)
        idx_no_kernel = dist_ker_centers > thr_kernel

        idx_candidates = idx_close * idx_no_kernel
        # candidates = torch.unique(all_traj[idx_candidates], dim=0)
        candidates = all_traj[idx_candidates]
        return candidates


# @torch.jit.script
def eval_policy(rbf_val: torch.Tensor,
                alphas: torch.Tensor):
    # Evaluate policy at q
    kernel_value = torch.sum(alphas * rbf_val, 1)
    return kernel_value


def eval_rbf(q: torch.Tensor,
             mu: torch.Tensor,
             sigma: torch.Tensor):
    # Evaluate policy at q
    # alphas: weights of RBF kernels
    # mu: centers of RBF kernels
    # sigma: standard deviations of RBF kernels
    # q: state
    # return: perturbation along tangent directions
    numerator = torch.norm(q[:, None, :] - mu, 2, 2, keepdim=True) ** 2
    denominator = 2 * sigma.unsqueeze(2) ** 2
    exp_term = torch.exp(-numerator / denominator)
    return exp_term


def check_traj_for_kernels(all_traj, closests_dist_all, kernel_val_all, thr_dist, thr_kernel):
    # Check if a trajectory has encountered an obstacle
    # If yes, output this position as potential kernel center
    # all_traj: all trajectories
    # closests_dist_all: distance to closest obstacle for each point of each trajectory
    # kernel_val_all: value of RBF kernels for each point of each trajectory
    # return: potential kernel centers
    kernel_val_all[:, -1] += 10
    idx_close = closests_dist_all < thr_dist
    kernel_val_all_sum = torch.sum(kernel_val_all, 2)
    idx_no_kernel = kernel_val_all_sum < thr_kernel
    if kernel_val_all.shape[-1]>0:
        max_ker_val, _ = torch.max(kernel_val_all, -1)
    else:
        max_ker_val = idx_close*0
    idx_no_kernel = max_ker_val < thr_kernel

    idx_candidates = idx_close * idx_no_kernel
    # candidates = torch.unique(all_traj[idx_candidates], dim=0)
    candidates = all_traj[idx_candidates]
    return candidates


if __name__ == '__main__':
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        dof = 2
        n_traj = 50
        tens_args = {'device': 'cuda:0', 'dtype': torch.float32}
        P = TensorPolicyMPPI(n_traj, dof, tens_args)
        t0 = time.time()
        torch.manual_seed(0)
        for i in range(100):
            P.add_kernel(torch.tensor([i / 100, i / 100], **tens_args))
            P.sample_policy()
            w_tmp = torch.arange(0, 1, 1 / n_traj, **tens_args)
            w_tmp = (w_tmp / sum(w_tmp)).flip(0)
            P.update_policy(w_tmp, 0.1)
        print(P.mu_c.shape)
        print(P.sigma_c.shape)
        print(P.alpha_c.shape)
        print(time.time() - t0, 'sec')

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
