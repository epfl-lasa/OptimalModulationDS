class Cost:
    def __init__(self, q_f):
        self.qf = q_f
        self.COLL_WEIGHT = 500

    def evaluate_costs(self, all_traj, closest_dist_all):
        goal_cost = self.goal_cost(all_traj[:, -1, :], self.qf)
        collision_cost = 500 * self.collision_cost(closest_dist_all)
        joint_limits_cost = 500 * self.joint_limits_cost(all_traj)
        stagnation_cost = 100 * goal_cost * self.stagnation_cost(all_traj)
        total_cost = goal_cost + collision_cost + joint_limits_cost + stagnation_cost
        return total_cost

    def goal_cost(self, traj_end, qf):
        return (traj_end - qf).norm(2, dim=1)

    def collision_cost(self, closest_dist_all):
        return (closest_dist_all < 0).sum(dim=1)

    def joint_limits_cost(self, all_traj):
        mask = (all_traj < -3.1415).sum(dim=1) + (all_traj > 3.1415).sum(dim=1)
        mask = mask.sum(dim=1)
        return (mask > 0) + 0

    def stagnation_cost(self, all_traj):
        dist = (all_traj[:, 0, :] - all_traj[:, -1, :]).norm(2, dim=1)
        return (1/dist).nan_to_num(0)