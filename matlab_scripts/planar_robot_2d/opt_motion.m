% point stabilization + Multiple shooting + Runge Kutta
clear all
close all force
clc
addpath('/home/michael/Documents/EPFL/Projects/casadi-linux-matlabR2014b-v3.5.5')
addpath('../')

import casadi.*

%% obstacle
load('data/net50_pos_thr.mat')
[y_f, dy_f] = tanhNN(net);
%[val, grad] = getGradAnalytical(y_f, dy_f, x);

%% robot
r = [4 4 0];
d = [0 0 0];
alpha = [0 0 0];
base = eye(4);
k1 = 0.9; k2 = 0.9;
q_min = [-k1*pi, -k2*pi, -10,-10];
q_max = [k1*pi, k2*pi, 10, 10];

%% PROBLEM SET UP
dt = 0.1; %[s]
H = 10; % prediction horizon
N_ITER = 150;
u1_max = pi; u1_min = -u1_max;

x1 = SX.sym('x1'); x2 = SX.sym('x2');
state = [x1;x2]; n_s = length(state);
x1_max = q_max(1); x1_min = q_min(1);
x2_max = q_max(2); x2_min = q_min(2);

u1 = SX.sym('u1'); u2 = SX.sym('u2');
controls = [u1]; n_c = length(controls);
A = [-5 0.2; 0.6 -1];
%A = -eye(2);
attr = [-0.8*pi; -0.5*pi];
attr = [-1.25; -1.1];
init_pos = [1.75; 1.8];

obs_pos = [9; 0];
obs_r = 3;

%rhs = A*(states - attr) + controls;
rotm = @(ang)[cos(ang) sin(ang);
             -sin(ang) cos(ang)];
rhs = rotm(controls(1)) * A*(state - attr)/max(0.5,norm(state - attr));

%rhs = [v*cos(theta);v*sin(theta);omega]; % system r.h.s

f = Function('f',{state,controls},{rhs}); % (nonlinear) mapping xdot=f(x,u)
U = SX.sym('U',n_c,H); % Decision variables (controls)
P = SX.sym('P',n_s + n_s); % parameters (initial and reference state)

X = SX.sym('X',n_s,(H+1));
% A vector that represents the states over the optimization problem.

obj = 0; % Objective function
g = [];  % constraints vector

W_SL = 0.1*eye(n_s,n_s);  % weighing matrix for states in lagrangian term
W_SM = 10*H*eye(n_s,n_s);  % weighing matrix for states in mayer term

W_C = eye(n_c,n_c);  % weighing matrix for controls
W_CR = 10*eye(n_c,n_c);  % weighing matrix for control rates

st  = X(:,1); % initial state
g = [g;st-P(1:2)]; % initial condition constraints
for k = 1:H
    st = X(:,k);  con = U(:,k);
    %objectove function
    if k == H % mayer term
        obj = obj+(st-P(3:4))'*W_SL*(st-P(3:4)); % final goal reaching
    else % lagrange term
        obj = obj+(st-P(3:4))'*W_SM*(st-P(3:4)); % final goal reaching
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

args.ubx(1:n_s:n_s*(H+1),1) = x1_max; %state x upper bound
args.ubx(2:n_s:n_s*(H+1),1) = x2_max; %state y upper bound

args.lbx(n_s*(H+1)+1:n_c:n_s*(H+1)+n_c*H,1) = u1_min; %u1 lower bound
%args.lbx(n_s*(N+1)+2:n_c:n_s*(N+1)+n_c*N,1) = u2_min; %u2 lower bound

args.ubx(n_s*(H+1)+1:n_c:n_s*(H+1)+n_c*H,1) = u1_max; %u1 upper bound
%args.ubx(n_s*(N+1)+2:n_c:n_s*(N+1)+n_c*N,1) = u2_max; %u2 upper bound



%% THE SIMULATION LOOP STARTS HERE
%-------------------------------------------
t0 = 0;
x0 = init_pos;
%x0 = [0.8*pi ; 0];    % initial condition.
x_ref = attr;% Reference posture.
x_opt(:,1) = x0; % xx contains the history of states
t(1) = t0;

u0 = -pi/2+zeros(H,n_c)+0*(rand(H,n_c)-0.5);        % two control inputs for each robot
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
        pause(1)
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
stairs(u_opt(:,1),'LineWidth',2);
%stairs(u_opt(:,2),'LineWidth',2);
legend('u1','u2')

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
