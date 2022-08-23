% point stabilization + Multiple shooting + Runge Kutta
clear all
close all
clc
addpath('/home/michael/Documents/EPFL/Projects/casadi-linux-matlabR2014b-v3.5.5')
import casadi.*
%% PROBLEM SET UP
dt = 0.3; %[s]
N = 20; % prediction horizon

u1_max = 10; u1_min = -u1_max;
u2_max = 10; u2_min = -u2_max;

x1 = SX.sym('x1'); x2 = SX.sym('x2');
states = [x1;x2]; n_s = length(states);

u1 = SX.sym('u1'); u2 = SX.sym('u2');
controls = [u1;u2]; n_c = length(controls);
A = [-1 0; 0 -1];
attr = [0; 0];
rhs = A*(states - attr) + controls;
%rhs = [v*cos(theta);v*sin(theta);omega]; % system r.h.s

f = Function('f',{states,controls},{rhs}); % (nonlinear) mapping xdot=f(x,u)
U = SX.sym('U',n_c,N); % Decision variables (controls)
P = SX.sym('P',n_s + n_s); % parameters (initial and reference state)

X = SX.sym('X',n_s,(N+1));
% A vector that represents the states over the optimization problem.

obj = 0; % Objective function
g = [];  % constraints vector

Q = eye(n_s,n_s);  % weighing matrix for states
R = 0.05*eye(n_c,n_c);  % weighing matrix for controls
st  = X(:,1); % initial state
g = [g;st-P(1:2)]; % initial condition constraints
for k = 1:N
    st = X(:,k);  con = U(:,k);
    obj = obj+(st-P(3:4))'*Q*(st-P(3:4)) + con'*R*con; % calculate obj
    st_next = X(:,k+1); %symbolic variable for next state
    k1 = f(st, con);   
    k2 = f(st + dt/2*k1, con);
    k3 = f(st + dt/2*k2, con); 
    k4 = f(st + dt*k3, con); 
    st_next_RK4=st +dt/6*(k1 +2*k2 +2*k3 +k4); 
    g = [g;st_next-st_next_RK4]; % multiple shooting state constraints 
end
% make the decision variable one column  vector
OPT_variables = [reshape(X,n_s*(N+1),1);reshape(U,n_c*N,1)];

nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

opts = struct;
opts.ipopt.max_iter = 2000;
opts.ipopt.print_level =0;%0,3
opts.print_time = 0;
opts.ipopt.acceptable_tol =1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

solver = nlpsol('solver', 'ipopt', nlp_prob,opts);

args = struct;

args.lbg(1:n_s*(N+1)) = 0;  % -1e-20  % Equality constraints for RK-state
args.ubg(1:n_s*(N+1)) = 0;  % 1e-20   % Equality constraints for RK-state

args.lbx(1:n_s:n_s*(N+1),1) = -10; %state x lower bound
args.lbx(2:n_s:n_s*(N+1),1) = -10; %state y lower bound

args.ubx(1:n_s:n_s*(N+1),1) = 10; %state x upper bound
args.ubx(2:n_s:n_s*(N+1),1) = 10; %state y upper bound

args.lbx(n_s*(N+1)+1:n_c:n_s*(N+1)+n_c*N,1) = u1_min; %u1 lower bound
args.lbx(n_s*(N+1)+2:n_c:n_s*(N+1)+n_c*N,1) = u2_min; %u2 lower bound

args.ubx(n_s*(N+1)+1:n_c:n_s*(N+1)+n_c*N,1) = u1_max; %u1 upper bound
args.ubx(n_s*(N+1)+2:n_c:n_s*(N+1)+n_c*N,1) = u2_max; %u2 upper bound



%% THE SIMULATION LOOP STARTS HERE
%-------------------------------------------
t0 = 0;
x0 = [-8 ; 5];    % initial condition.
xs = [8; 8]; % Reference posture.

xx(:,1) = x0; % xx contains the history of states
t(1) = t0;

u0 = zeros(N,2);        % two control inputs for each robot
X0 = repmat(x0,1,N+1)'; % initialization of the states decision variables

max_sim_time = 5; % Maximum simulation time

% Start MPC
mpciter = 0;
all_traj = [];
all_u=[];

% the main simulaton loop... it works as long as the error is greater
% than 10^-6 and the number of mpc steps is less than its maximum
% value.
main_loop = tic;
while(mpciter < max_sim_time / dt)
    args.p   = [x0;xs]; % set the values of the parameters vector
    % initial value of the optimization variables
    args.x0  = [reshape(X0',n_s*(N+1),1);reshape(u0',n_c*N,1)];
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);
    u_traj = reshape(full(sol.x(n_s*(N+1)+1:end))',n_c,N)'; % get controls from the solution
    all_u= [all_u ; u_traj(1,:)];

    x_traj = reshape(full(sol.x(1:n_s*(N+1)))',n_s,N+1)'; % get solution trajectory
    all_traj(:,1:n_s,mpciter+1)= x_traj; % store solution trajectory

    t(mpciter+1) = t0;
    % Apply the control and shift the solution
    [t0, x0, u0] = shift(dt, t0, x0, u_traj,f);
    xx(:,mpciter+2) = x0;
    X0 = x_traj;
    % Shift trajectory to initialize the next step
    X0 = [X0(2:end,:);X0(end,:)];
    mpciter = mpciter + 1;
    disp(mpciter)
end
main_loop_time = toc(main_loop);
ss_error = norm((x0-xs),2);
average_mpc_time = main_loop_time/(mpciter+1)

%%
figure()
plot(xx(1,:), xx(2,:),'LineWidth',2)
hold on
plot(xx(1,1), xx(2,1),'r*')
plot(xx(1,end), xx(2,end),'b*')

xaxis([-10,10])
yaxis([-10,10])
figure()
stairs(all_u(:,1),'LineWidth',2)
hold on
stairs(all_u(:,2),'LineWidth',2)
legend('u1','u2')


function [t0, x0, u0] = shift(dt, t0, x0, u,f)
    st = x0;
    con = u(1,:)';
    f_value = f(st,con);
    st = st + (dt*f_value);
    x0 = full(st);
    
    t0 = t0 + dt;
    u0 = [u(2:size(u,1),:);u(size(u,1),:)];
end
