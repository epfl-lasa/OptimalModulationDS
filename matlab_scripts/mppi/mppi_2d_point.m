clc
clear all
close all force
addpath('../planar_robot_7d/')
rng('default')
%% Constants and Parameters
N_ITER = 1000;
H = 30;
D = 2;
SIGMA = [3.1];
N_POL = size(SIGMA,2);
MU_ARR = zeros(D, H, N_POL);

dT = 0.1;
N_TRAJ = 50;
pos_init = zeros(D, 1);
pos_goal = pos_init+6;
gamma_vec = [0.98.^linspace(1,H-1,H-1), 1.02^H];
beta = 0.9;
mu_alpha = 0.9;
%% lambdas
get_norm_samples = @(MU_ARR, SIGMA, N_TRAJ)...
                        normrnd(repmat(MU_ARR,[1,1,N_TRAJ]), SIGMA);
INT_MAT = tril(ones(H));
get_rollout = @(pos_init, u_sampl, dT) pos_init + ...
              pagetranspose(pagemtimes(dT*INT_MAT,pagetranspose(u_sampl)));

calc_reaching_cost = @(rollout, goal) squeeze(vecnorm(rollout - goal,2,1))';
obs_dist = @(rollout, obs_pos, obs_r) squeeze(vecnorm(rollout - obs_pos,2,1))'-obs_r;
w_fun = @(cost) exp(-1/beta * sum(gamma_vec .* cost,2));
%mu_upd_fun = @(mu, w, u) (1-mu_alpha)*mu +
%% plot preparation
f = figure(1);
ax_h = axes(f);
axis equal
set(gcf,'color','w');
title('MPPI example')
xlabel('x_1')
ylabel('x_2')

hold on
ax_h.XLim = [-1,10];
ax_h.YLim = [-1,10];
plot(pos_init(1),pos_init(2),'b*','MarkerSize', 5, 'LineWidth', 3)
plot(pos_goal(1),pos_goal(2),'r*','MarkerSize', 5, 'LineWidth', 3)
r_h_arr = zeros(N_POL, N_TRAJ);
for i = 1:1:N_POL
    for j = 1:1:N_TRAJ
        r_h = plot(ax_h,zeros(1,H), zeros(1,H), 'LineWidth', 2);
        r_h.Color=[1-1/i,1-1/i,1/i,0.5];
        r_h_arr(i, j) = r_h;
    end
end
best_traj_h = plot(ax_h,zeros(1,H), zeros(1,H));
best_traj_h.Color=[0,1,0.5,1];
best_traj_h.LineWidth = 2;
cur_pos_h = plot(0,0,'r*');
%% MPPI

cur_pos = pos_init;
cur_vel = pos_init*0;
for i = 1:1:N_ITER
    best_cost_iter = 1e10;
    for j = 1:1:N_POL
        tic
        u = get_norm_samples(MU_ARR(:,:,j),SIGMA(j),N_TRAJ);
        u(:,:,end) = u(:,:,end)*0;
        %inject slowdown (for acceleration control)
%         ss = 5;
%         u(:,1:ss,end) = u(:,1:ss,end) - cur_vel/dT/ss;

        v_rollout = get_rollout(cur_vel, u, dT);
        rollout = get_rollout(cur_pos, v_rollout, dT);
        rollout = v_rollout;
        cost_p = calc_reaching_cost(rollout, pos_goal);
        cost_v = 0*calc_reaching_cost(v_rollout, cur_vel*0);
        
        %d1 = obs_dist(rollout, [3; 3], 2);
        %d1(d1>0) = 0;
        %d1(d1<0) = 500;
        cost = cost_p+cost_v;
        w = w_fun(cost);
        w = w/sum(w);
        [best_cost,best_idx] = max(w);
        w_tens = reshape(repmat(w,[1,H])', [1, H ,N_TRAJ]);
        MU_ARR(:,:,j) = (1-mu_alpha)*MU_ARR(:,:,j)+mu_alpha*sum(w_tens.*u,3);
        if best_cost < best_cost_iter
            cur_pos = rollout(:,1,best_idx);
            cur_vel = v_rollout(:,1,best_idx);
            cur_pos_h.XData = cur_pos(1);
            cur_pos_h.YData = cur_pos(2);
            best_cost_iter = best_cost;
            best_traj_h.XData = rollout(1,:,best_idx);
            best_traj_h.YData = rollout(2,:,best_idx);
        end
        for i_traj = 1:1:N_TRAJ
            h_tmp = handle(r_h_arr(j, i_traj));
            h_tmp.XData = rollout(1,:,i_traj);
            h_tmp.YData = rollout(2,:,i_traj);
        end

        toc
        pause(0.1)
    end
end













