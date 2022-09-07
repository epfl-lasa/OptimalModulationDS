% point stabilization + Multiple shooting + Runge Kutta
clear all
close all force
clc
addpath('/home/michael/Documents/EPFL/Projects/casadi-linux-matlabR2014b-v3.5.5')
addpath('../')

import casadi.*

%% obstacle
load('data/net128_pos.mat')
[y_f, dy_f] = tanhNN(net);
%[val, grad] = getGradAnalytical(y_f, dy_f, x);

%% robot
DOF = 7;
l1=1;
%dh elements
r = [repmat(l1,[1, DOF]), 0];
d = 0*r;
alpha = 0*r;
base = eye(4);
k_lim = 0.9;
q_min = [repmat(-k_lim*pi,[1,7]), -10, -10];
q_max = [repmat(k_lim*pi,[1,7]), 10, 10];

%% PROBLEM SET UP
dt = 0.05; %[s]
H = 10; % prediction horizon
N_ITER = 150;
u1_max = 1; u1_min = -1;

x1 = SX.sym('x1'); x2 = SX.sym('x2');
x3 = SX.sym('x3'); x4 = SX.sym('x4');
x5 = SX.sym('x5'); x6 = SX.sym('x6');
x7 = SX.sym('x7'); 
state = [x1;x2;x3;x4;x5;x6;x7]; n_s = length(state);
x1_max = q_max(1); x1_min = q_min(1);
x2_max = q_max(2); x2_min = q_min(2);
x3_max = q_max(3); x3_min = q_min(3);
x4_max = q_max(4); x4_min = q_min(4);
x5_max = q_max(5); x5_min = q_min(5);
x6_max = q_max(6); x6_min = q_min(6);
x7_max = q_max(7); x7_min = q_min(7);

u1 = SX.sym('u1'); u2 = SX.sym('u2');
u3 = SX.sym('u3'); u4 = SX.sym('u4');
u5 = SX.sym('u5'); u6 = SX.sym('u6');
u7 = SX.sym('u7');

controls = [u1; u2; u3; u4; u5; u6; u7]; n_c = length(controls);
A = -1*eye(7);
attr = repmat(-0.1*pi,[1,7])';
init_pos = repmat(0.1*pi,[1,7])';

obs_pos = [7; 3];
obs_r = 1;

%rhs = A*(state - attr)./abs(state - attr) + controls;
rhs = A*(state - attr) + controls;

% rotm = @(ang)[cos(ang) sin(ang);
%              -sin(ang) cos(ang)];
% rhs = rotm(controls(1)) * A*(state - attr)/max(0.5,norm(state - attr));

%rhs = [v*cos(theta);v*sin(theta);omega]; % system r.h.s

f = Function('f',{state,controls},{rhs}); % (nonlinear) mapping xdot=f(x,u)
U = SX.sym('U',n_c,H); % Decision variables (controls)
P = SX.sym('P',n_s + n_s); % parameters (initial and reference state)

X = SX.sym('X',n_s,(H+1));
% A vector that represents the states over the optimization problem.

obj = 0; % Objective function
g = [];  % constraints vector

W_SL = 0.01*eye(n_s,n_s);  % weighing matrix for states in lagrangian term
W_SM = 10*H*eye(n_s,n_s);  % weighing matrix for states in mayer term

W_C = 0*0.001*eye(n_c,n_c);  % weighing matrix for controls
W_CR = 0*0.1*eye(n_c,n_c);  % weighing matrix for control rates

st  = X(:,1); % initial state
g = [g;st-P(1:DOF)]; % initial condition constraints
for k = 1:H
    st = X(:,k);  con = U(:,k);
    %objective function
    if k == H % mayer term
        obj = obj+(st-P(DOF+1:end))'*W_SL*(st-P(DOF+1:end)); % final goal reaching
    else % lagrange term
        obj = obj+(st-P(DOF+1:end))'*W_SM*(st-P(DOF+1:end)); % final goal reaching
        obj = obj +  con'*W_C*con; % penalize large U
        con_rate = U(:,k+1) - U(:,k);
        obj = obj + con_rate'*W_CR*con_rate; %penalize rate of U
    end
    
    st_next = X(:,k+1); %symbolic variable for next state
    k1 = f(st, con);   
    k2 = f(st + dt/2*k1, con);
    k3 = f(st + dt/2*k2, con); 
    k4 = f(st + dt*k3, con); 
    st_next_RK4=st +dt/6*(k1 +2*k2 +2*k3 +k4); 
    g = [g;st_next-st_next_RK4]; % multiple shooting state constraints 
end

% Add constraints for collision avoidance
%obs_x = 0; obs_y = 0; obs_r = 2;
thr = 0.001;
for k = 1:H+1   % box constraints due to the map margins
    %g = [g ; sqrt((X(1,k)-obs_x)^2+(X(2,k)-obs_y)^2)-obs_r];
    inp = [X(:,k); obs_pos];
    pos_inp = [inp; sin(inp); cos(inp)];
    g = [g ; y_f(pos_inp)-obs_r];
end

% make the decision variable one column  vector
OPT_variables = [reshape(X,n_s*(H+1),1);reshape(U,n_c*H,1)];

nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

opts = struct;
opts.ipopt.max_iter = 3000;
opts.ipopt.print_level =3;%0,3
opts.print_time = 0;
opts.ipopt.acceptable_tol =1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

solver = nlpsol('solver', 'ipopt', nlp_prob,opts);

args = struct;

args.lbg(1:n_s*(H+1)) = 0;  % -1e-20  % Equality constraints for RK-state
args.ubg(1:n_s*(H+1)) = 0;  % 1e-20   % Equality constraints for RK-state
args.lbg(n_s*(H+1)+1 : n_s*(H+1)+ (H+1)) = thr; % inequality constraints for obstacle
args.ubg(n_s*(H+1)+1 : n_s*(H+1)+ (H+1)) = +inf; % inequality constraints for obstacle

args.lbx(1:n_s:n_s*(H+1),1) = x1_min; %state x lower bound
args.lbx(2:n_s:n_s*(H+1),1) = x2_min; %state y lower bound
args.lbx(3:n_s:n_s*(H+1),1) = x3_min; %state y lower bound
args.lbx(4:n_s:n_s*(H+1),1) = x4_min; %state y lower bound
args.lbx(5:n_s:n_s*(H+1),1) = x5_min; %state y lower bound
args.lbx(6:n_s:n_s*(H+1),1) = x6_min; %state y lower bound
args.lbx(7:n_s:n_s*(H+1),1) = x7_min; %state y lower bound

args.ubx(1:n_s:n_s*(H+1),1) = x1_max; %state x upper bound
args.ubx(2:n_s:n_s*(H+1),1) = x2_max; %state y upper bound
args.ubx(3:n_s:n_s*(H+1),1) = x3_max; %state y upper bound
args.ubx(4:n_s:n_s*(H+1),1) = x4_max; %state y upper bound
args.ubx(5:n_s:n_s*(H+1),1) = x5_max; %state y upper bound
args.ubx(6:n_s:n_s*(H+1),1) = x6_max; %state y upper bound
args.ubx(7:n_s:n_s*(H+1),1) = x7_max; %state y upper bound

args.lbx(n_s*(H+1)+1:n_c:n_s*(H+1)+n_c*H,1) = u1_min; %u1 lower bound
args.lbx(n_s*(H+1)+2:n_c:n_s*(H+1)+n_c*H,1) = u1_min; %u1 lower bound
args.lbx(n_s*(H+1)+3:n_c:n_s*(H+1)+n_c*H,1) = u1_min; %u1 lower bound
args.lbx(n_s*(H+1)+4:n_c:n_s*(H+1)+n_c*H,1) = u1_min; %u1 lower bound
args.lbx(n_s*(H+1)+5:n_c:n_s*(H+1)+n_c*H,1) = u1_min; %u1 lower bound
args.lbx(n_s*(H+1)+6:n_c:n_s*(H+1)+n_c*H,1) = u1_min; %u1 lower bound
args.lbx(n_s*(H+1)+7:n_c:n_s*(H+1)+n_c*H,1) = u1_min; %u1 lower bound

args.ubx(n_s*(H+1)+1:n_c:n_s*(H+1)+n_c*H,1) = u1_max; %u1 upper bound
args.ubx(n_s*(H+1)+2:n_c:n_s*(H+1)+n_c*H,1) = u1_max; %u2 upper bound
args.ubx(n_s*(H+1)+3:n_c:n_s*(H+1)+n_c*H,1) = u1_max; %u2 upper bound
args.ubx(n_s*(H+1)+4:n_c:n_s*(H+1)+n_c*H,1) = u1_max; %u2 upper bound
args.ubx(n_s*(H+1)+5:n_c:n_s*(H+1)+n_c*H,1) = u1_max; %u2 upper bound
args.ubx(n_s*(H+1)+6:n_c:n_s*(H+1)+n_c*H,1) = u1_max; %u2 upper bound
args.ubx(n_s*(H+1)+7:n_c:n_s*(H+1)+n_c*H,1) = u1_max; %u2 upper bound



%% THE SIMULATION LOOP STARTS HERE
%-------------------------------------------
t0 = 0;
x0 = init_pos;
%x0 = [0.8*pi ; 0];    % initial condition.
x_ref = attr;% Reference posture.
x_opt(:,1) = x0; % xx contains the history of states
t(1) = t0;

u0 = -pi/2*0+zeros(H,n_c)+0*(rand(H,n_c)-0.5);        % two control inputs for each robot
u0(1,:) = zeros(1,n_c);
X0 = repmat(x0,1,H+1)'; % initialization of the states decision variables


% Start MPC
mpciter = 0;
%all_traj = [];
u_opt=[];

% the main simulaton loop... it works as long as the error is greater
% than 10^-6 and the number of mpc steps is less than its maximum
% value.
main_loop = tic;
while(mpciter < N_ITER && norm(x0-x_ref)>1e-1)
    args.p   = [x0;x_ref]; % set the values of the parameters vector
    % initial value of the optimization variables
    args.x0  = [reshape(X0',n_s*(H+1),1);reshape(u0',n_c*H,1)];
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);
    u_traj = reshape(full(sol.x(n_s*(H+1)+1:end))',n_c,H)'; % get controls from the solution
    u_opt= [u_opt ; u_traj(1,:)];
    u_traj(1,:)
    x_traj = reshape(full(sol.x(1:n_s*(H+1)))',n_s,H+1)'; % get solution trajectory
    all_traj(:,1:n_s,mpciter+1)= x_traj; % store solution trajectory

    t(mpciter+1) = t0;
    % Apply the control and shift the solution
    %use RK
    t01 = t0+dt;
    x01 = x_traj(2,:)';
    u01 = [u_traj(2:end,:);u_traj(end,:)];

    [t0, x0, u0] = shift(dt, t0, x0, u_traj,f); %use euler
    x_opt(:,mpciter+2) = x0;
    X0 = x_traj;
    % Shift trajectory to initialize the next step
    X0 = [X0(2:end,:);X0(end,:)];
    mpciter = mpciter + 1;
    disp(mpciter)
    %plotting part
    if mpciter == 1
        prepare_plot_planar_jspace
    end
    online_plot_planar_jspace
    %pause(0.5)
end
main_loop_time = toc(main_loop);
x0'
x_ref'
ss_error = norm((x0-x_ref),2)
average_mpc_time = main_loop_time/(mpciter)

%% control stairs
figure(2)
title('Control input')
hold on
for i = 1:1:DOF
    stairs(u_opt(:,i),'LineWidth',2);
end
%stairs(u_opt(:,2),'LineWidth',2);
legend('u1','u2','u3','u5','u5','u6','u7')

%% helper functions
function [t0, x0, u0] = shift(dt, t0, x0, u,f)
    st = x0;
    con = u(1,:)';
    f_value = f(st,con);
    st = st + (dt*f_value);
    x0 = full(st);
    
    t0 = t0 + dt;
    u0 = [u(2:size(u,1),:);u(size(u,1),:)];
end
