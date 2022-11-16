clc
clear all
close all force
addpath('../planar_robot_2d/')
rng('default')

%% obstacle
load('../planar_robot_2d/data/net50_pos_thr.mat')
[y_f, dy_f] = tanhNN(net);
obs_pos = [5 3]';
obs_r = 1;
%% DS
A = [-10 0;
      0 -1];
f = @(A, x, x_goal) A * (x - x_goal);
M = @(n_v, tau_v, l) [n_v tau_v] * diag(l) * [n_v tau_v]';

%% robot
r = [4 4 0];
d = [0 0 0];
alpha = [0 0 0];
base = eye(4);
k1 = 0.9; k2 = 0.9;
q_min = [-k1*pi, -k2*pi, -10,-10];
q_max = [k1*pi, k2*pi, 10, 10];
box = [q_min(1:2); q_max(1:2)];
u_box = 2*[-1, 1; -1 1]';
pos_init = [0.5*pi, 0.5*pi]';
pos_goal = [-0.5*pi, -0.5*pi]';

%figure
f_anim = figure('Name','Animation','Position',[100 100 1400 400]);
ax_anim = subplot(1,2,1);
axis equal
title('Planar robot')
xlabel('x, m')
ylabel('y, m')

axes(ax_anim)
ax_anim.XLim = [-12; 12];
ax_anim.YLim = [-12; 12];
hold on
robot_h = create_r(ax_anim,pos_init,r,d,alpha,base);

[xc, yc] = circle_pts(obs_pos(1),obs_pos(2),obs_r-0.01);
crc_h = plot(ax_anim, xc, yc, 'r-','LineWidth',1.5);

%% Constants and Parameters
N_ITER = 100000;
H = 50;
D = 2;
SIGMA = [1, 0.5, 0.1];
N_POL = size(SIGMA,2);
MU_ARR = zeros(D, H, N_POL);
N_TRAJ = 30;
gamma_vec = flip([0.98.^linspace(1,H-1,H-1), 0.98^H]);
beta = 0.9;
mu_alpha = 0.99;
%% lambdas
get_norm_samples = @(MU_ARR, SIGMA, N_TRAJ)...
                        normrnd(repmat(MU_ARR,[1,1,N_TRAJ]), SIGMA);
INT_MAT = tril(ones(H));
get_rollout = @(pos_init, u_sampl, dT) pos_init + ...
              pagetranspose(pagemtimes(dT*INT_MAT,pagetranspose(u_sampl)));

calc_reaching_cost = @(rollout, goal) squeeze(vecnorm(rollout - goal,2,1))';
obs_dist = @(rollout, obs_pos, obs_r) squeeze(vecnorm(rollout - obs_pos,2,1))'-obs_r;
w_fun = @(cost) exp(-1/beta * sum(gamma_vec .* cost,2));

%% plot preparation
ax_proj = subplot(1,2,2);
axis equal
title('2d MPPI vis')
xlabel('q1')
ylabel('q2')
hold on
%joint-space
x1_span=linspace(box(1,1),box(2,1),50); 
x2_span=linspace(box(1,2),box(2,2),50); 
[X1_mg,X2_mg] = meshgrid(x1_span, x2_span);
x=[X1_mg(:) X2_mg(:)]';

%nominal ds
% obstacle
inp = [x ; repmat(obs_pos,[1,length(x)])];
val = y_f([inp; sin(inp); cos(inp)]);
Z_mg = reshape(val,size(X1_mg));
%distance heatmap and contour
[~, obs_h] = contourf(ax_proj, X1_mg,X2_mg,Z_mg,100,'LineStyle','none');
[~, obs2_h] = contour(ax_proj, X1_mg,X2_mg,Z_mg,[obs_r,obs_r+0.01],'LineStyle','-','LineColor','k','LineWidth',2);

ax_proj.XLim = box(:,1);
ax_proj.YLim = box(:,2);
plot(pos_init(1),pos_init(2),'b*')
plot(pos_goal(1),pos_goal(2),'r*')
r_h_arr = zeros(N_TRAJ);
for i = 1:1:N_TRAJ
    r_h = plot(ax_proj,zeros(1,H), zeros(1,H));
    r_h.Color=[1-1/i,1-1/i,1/i,0.5];
    r_h_arr(i) = r_h;
end
best_traj_h = plot(ax_proj,zeros(1,H), zeros(1,H));
best_traj_h.Color=[0,1,0.5,1];
best_traj_h.LineWidth = 2;
cur_pos_h = plot(0,0,'r*');
%% MPPI
dT = 0.2;
max_vel = 0.1;
cur_pos = pos_init;
cur_vel = pos_init*0;
for i = 1:1:N_ITER
    %calculate ds
    inp = [cur_pos; obs_pos];
    inp_pe = [inp; sin(inp); cos(inp)];
    dist = y_f(inp_pe)- obs_r;
    thr = 0;
    if dist <=thr+0.01
        aa = 10;
    end
    n_v = dy_f(inp_pe)';
    dq1 = n_v(1:2);
    dsin = n_v(5:6);
    dcos = n_v(9:10);
    dfdq = dq1 + dsin.*cos(cur_pos) - dcos.*sin(cur_pos); 
    n_v = n_v(1:2)/norm(n_v(1:2));
    n_v = dfdq/norm(dfdq);
    v = [cur_pos cur_pos+0.2*n_v];
    %plot(v(1,:), v(2,:),'.-','Color','r')
    tau_v = [n_v(2); -n_v(1)];
    %calculate velocity
    g = max(1e-8, dist-thr+1);
    l_n = max(0, 1 - 1/g);
    l_tau = min(1, 1 + 1/g);
    E = [n_v tau_v];
    D = [l_n 0;
         0 l_tau];
    %D = [1 0;
    %     0 1];
    nominal_vel = f(A, cur_pos, pos_goal);
    if norm(nominal_vel)>max_vel
        nominal_vel = max_vel*nominal_vel/norm(nominal_vel);
    end

    cur_vel = E*D*E' * nominal_vel
    if norm(cur_vel)>max_vel
        cur_vel = max_vel*cur_vel/norm(cur_vel);
    end

    cur_pos = cur_pos + dT * cur_vel;
    %plots
    %move robot
    move_r(robot_h,cur_pos,r,d,alpha,base);
    %move 2d proj
    cur_pos_h.XData = cur_pos(1);
    cur_pos_h.YData = cur_pos(2);
    %move mpc_traj
    for i_traj = 1:1:N_TRAJ
        h_tmp = handle(r_h_arr(i_traj));
        %h_tmp.XData = rollout(1,:,i_traj);
        %h_tmp.YData = rollout(2,:,i_traj);
    end
    pause(0.02)
end




%% functions
function cost = calc_lim_cost(traj, v_min, v_max)
    cost = zeros(size(traj,3), size(traj,2));
    min_tens = repmat(v_min',1,size(traj,2),size(traj,3));
    max_tens = repmat(v_max',1,size(traj,2),size(traj,3));
    cost = squeeze(any(traj<min_tens,1))' | squeeze(any(traj>max_tens,1))';
%     for i = 1:1:length(v_min)
%         cost = cost | squeeze(traj(i,:,:)<v_min(i) | traj(i,:,:)>v_max(i))';
%     end
    cost = double(cost);
end

function dst = calc_nn_dists(y_f, rollout, obs_pos, obs_r)
    r_tmp = reshape(rollout,size(rollout,1),[]);
    inp = [r_tmp; repmat(obs_pos,1,size(r_tmp,2))];
    dst = y_f([inp; sin(inp); cos(inp)])- obs_r;
    dst = reshape(dst,[],size(rollout,3))';
end

function cost = dist_cost(dst, thr0, thr1)
    cost = dst;
    idx_0 = dst > thr1; % no cost for above thr1
    idx_1 = dst < thr0; % positive cost for below thr0
    idx_smooth = ~(idx_0 | idx_1);
    cost(idx_0) = 0;
    cost(idx_1) = 1;
    cost(idx_smooth) = 1-cost(idx_smooth)./thr1;
end


function cost = calc_smooth_cost(traj)
    d_traj = diff(traj,1,2);
    cost = squeeze(vecnorm(d_traj,2,1));
    cost(end+1,:) = 0*cost(end,:);
    cost = 10*cost';
end


%robot and plotting
function pts = calc_fk(j_state,r,d,alpha,base)
    P = dh_fk(j_state,r,d,alpha,base);
    pts = zeros(3,3);
    for i = 1:1:3
        v = [0,0,0];
        R = P{i}(1:3,1:3);
        T = P{i}(1:3,4);
        p = v*R'+T';
        pts(i,:) = p;
    end
end

function handle = create_r(ax_h,j_state,r,d,alpha,base)
    pts = calc_fk(j_state,r,d,alpha,base);
    handle = plot(ax_h,pts(:,1),pts(:,2),'LineWidth',2,...
        'Marker','o','MarkerFaceColor','k','MarkerSize',4);
end

function move_r(r_handle,j_state,r,d,alpha,base)
    pts = calc_fk(j_state,r,d,alpha,base);
    r_handle.XData = pts(:,1);
    r_handle.YData = pts(:,2);
end

function [xc,yc] = circle_pts(x,y,r)
    hold on
    th = linspace(0,2*pi,50);
    xc = r * cos(th) + x;
    yc = r * sin(th) + y;
end





