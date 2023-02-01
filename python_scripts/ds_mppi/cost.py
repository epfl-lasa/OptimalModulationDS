from fk_num import *


class Cost:
    def __init__(self, q_f, dh_params):
        self.qf = q_f
        self.COLL_WEIGHT = 500
        self.dh_params = dh_params
        self.goal_fk = numeric_fk_model(self.qf, self.dh_params, 2)[0]
    def evaluate_costs(self, all_traj, closest_dist_all):
        goal_cost = self.goal_cost(all_traj[:, -1, :], self.qf)
        collision_cost = 500 * self.collision_cost(closest_dist_all)
        joint_limits_cost = 500 * self.joint_limits_cost(all_traj)
        stagnation_cost = 100 * goal_cost * self.stagnation_cost(all_traj)

        fk_cost = 10*self.fk_cost(all_traj[:, -1, :])

        total_cost = goal_cost + collision_cost + joint_limits_cost + stagnation_cost + fk_cost
        return total_cost

    def goal_cost(self, traj_end, qf):
        return (traj_end - qf).norm(2, dim=1)

    def fk_cost(self, traj_end):
        traj_end_fk = numeric_fk_model_vec(traj_end, self.dh_params, 2)[0]
        fk_diff = traj_end_fk[:, :, -1, :] - self.goal_fk[:, -1, :]
        fk_cost = fk_diff.norm(2, dim=2)[:, :].sum(dim=1)
        return fk_cost

    def collision_cost(self, closest_dist_all):
        return (closest_dist_all < 0).sum(dim=1)

    def joint_limits_cost(self, all_traj):
        mask = (all_traj < -3.1415).sum(dim=1) + (all_traj > 3.1415).sum(dim=1)
        mask = mask.sum(dim=1)
        return (mask > 0) + 0

    def stagnation_cost(self, all_traj):
        dist = (all_traj[:, 0, :] - all_traj[:, -1, :]).norm(2, dim=1)
        return (1/dist).nan_to_num(0)