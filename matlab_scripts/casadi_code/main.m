% point stabilization + Multiple shooting + Runge Kutta
clear all
close all
clc
addpath('/home/michael/Documents/EPFL/Projects/casadi-linux-matlabR2014b-v3.5.5')
import casadi.*
%% PROBLEM SET UP
dt = 0.05; %[s]
N = 20; % prediction horizon
N_ITER = 30;
u1_max = pi; u1_min = -u1_max;
u2_max = pi; u2_min = -u2_max;

x1 = SX.sym('x1'); x2 = SX.sym('x2');
states = [x1;x2]; n_s = length(states);

u1 = SX.sym('u1'); u2 = SX.sym('u2');
controls = [u1]; n_c = length(controls);
A = [-1 0; 0 -1];
attr = [8; 0];
%rhs = A*(states - attr) + controls;
rhs = [cos(controls(1)) sin(controls(1));
      -sin(controls(1)) cos(controls(1))] * A*(states - attr);

%rhs = [v*cos(theta);v*sin(theta);omega]; % system r.h.s

f = Function('f',{states,controls},{rhs}); % (nonlinear) mapping xdot=f(x,u)
U = SX.sym('U',n_c,N); % Decision variables (controls)
P = SX.sym('P',n_s + n_s); % parameters (initial and reference state)

X = SX.sym('X',n_s,(N+1));
% A vector that represents the states over the optimization problem.

obj = 0; % Objective function
g = [];  % constraints vector

Q = eye(n_s,n_s);  % weighing matrix for states
R = 0.1*eye(n_c,n_c);  % weighing matrix for controls
st  = X(:,1); % initial state
g = [g;st-P(1:2)]; % initial condition constraints
for k = 1:N
    st = X(:,k);  con = U(:,k);
    obj = obj + (st-P(3:4))'*Q*(st-P(3:4)) + con'*R*con; % calculate obj
    st_next = X(:,k+1); %symbolic variable for next state
    k1 = f(st, con);   
    k2 = f(st + dt/2*k1, con);
    k3 = f(st + dt/2*k2, con); 
    k4 = f(st + dt*k3, con); 
    st_next_RK4=st +dt/6*(k1 +2*k2 +2*k3 +k4); 
    g = [g;st_next-st_next_RK4]; % multiple shooting state constraints 
    if k == N
        obj = obj+(st-P(3:4))'*Q*(st-P(3:4)); % calculate obj
    end
end

% Add constraints for collision avoidance
obs_x = 0; 
obs_y = 0; 
obs_r = 2;
for k = 1:N+1   % box constraints due to the map margins
    g = [g ; sqrt((X(1,k)-obs_x)^2+(X(2,k)-obs_y)^2)-obs_r];
end

% make the decision variable one column  vector
OPT_variables = [reshape(X,n_s*(N+1),1);reshape(U,n_c*N,1)];

nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

opts = struct;
opts.ipopt.max_iter = 2000;
opts.ipopt.print_level =3;%0,3
opts.print_time = 0;
opts.ipopt.acceptable_tol =1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

solver = nlpsol('solver', 'ipopt', nlp_prob,opts);

args = struct;

args.lbg(1:n_s*(N+1)) = 0;  % -1e-20  % Equality constraints for RK-state
args.ubg(1:n_s*(N+1)) = 0;  % 1e-20   % Equality constraints for RK-state
args.lbg(n_s*(N+1)+1 : n_s*(N+1)+ (N+1)) = 0.5; % inequality constraints for obstacle
args.ubg(n_s*(N+1)+1 : n_s*(N+1)+ (N+1)) = +inf; % inequality constraints for obstacle

args.lbx(1:n_s:n_s*(N+1),1) = -10; %state x lower bound
args.lbx(2:n_s:n_s*(N+1),1) = -10; %state y lower bound

args.ubx(1:n_s:n_s*(N+1),1) = 10; %state x upper bound
args.ubx(2:n_s:n_s*(N+1),1) = 10; %state y upper bound

args.lbx(n_s*(N+1)+1:n_c:n_s*(N+1)+n_c*N,1) = u1_min; %u1 lower bound
%args.lbx(n_s*(N+1)+2:n_c:n_s*(N+1)+n_c*N,1) = u2_min; %u2 lower bound

args.ubx(n_s*(N+1)+1:n_c:n_s*(N+1)+n_c*N,1) = u1_max; %u1 upper bound
%args.ubx(n_s*(N+1)+2:n_c:n_s*(N+1)+n_c*N,1) = u2_max; %u2 upper bound



%% THE SIMULATION LOOP STARTS HERE
%-------------------------------------------
t0 = 0;
x0 = [-9 ; 0];    % initial condition.
x_ref = [8; 0]; % Reference posture.

x_opt(:,1) = x0; % xx contains the history of states
t(1) = t0;

u0 = zeros(N,n_c)+1000*(rand(N,n_c)-0.5);        % two control inputs for each robot
u0(1,:) = zeros(1,n_c);
X0 = repmat(x0,1,N+1)'; % initialization of the states decision variables


% Start MPC
mpciter = 0;
%all_traj = [];
u_opt=[];

% the main simulaton loop... it works as long as the error is greater
% than 10^-6 and the number of mpc steps is less than its maximum
% value.
main_loop = tic;
while(mpciter < N_ITER)
    args.p   = [x0;x_ref]; % set the values of the parameters vector
    % initial value of the optimization variables
    args.x0  = [reshape(X0',n_s*(N+1),1);reshape(u0',n_c*N,1)];
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);
    u_traj = reshape(full(sol.x(n_s*(N+1)+1:end))',n_c,N)'; % get controls from the solution
    u_opt= [u_opt ; u_traj(1,:)];

    x_traj = reshape(full(sol.x(1:n_s*(N+1)))',n_s,N+1)'; % get solution trajectory
    all_traj(:,1:n_s,mpciter+1)= x_traj; % store solution trajectory

    t(mpciter+1) = t0;
    % Apply the control and shift the solution
    [t0, x0, u0] = shift(dt, t0, x0, u_traj,f);
    x_opt(:,mpciter+2) = x0;
    X0 = x_traj;
    % Shift trajectory to initialize the next step
    X0 = [X0(2:end,:);X0(end,:)];
    mpciter = mpciter + 1;
    disp(mpciter)
end
main_loop_time = toc(main_loop);
x0'
x_ref'
ss_error = norm((x0-x_ref),2)
average_mpc_time = main_loop_time/(mpciter)

%% figures
%nominal ds
x_span=linspace(-10,10,50); 
y_span=linspace(-10,10,50); 
[X_mg,Y_mg] = meshgrid(x_span, y_span);
x=[X_mg(:) Y_mg(:)]';
x_dot = A*(x-attr);
U_nominal = reshape(x_dot(1,:), length(y_span), length(x_span));
V_nominal = reshape(x_dot(2,:), length(y_span), length(x_span));

% figure(1)
% hold on
% %nominal ds
% l = streamslice(X_mg,Y_mg,U_nominal,V_nominal);
% axis equal
% title('Nominal DS')
% circle([obs_x;obs_y],obs_r,'LineWidth',2)
% % obstacle
% [val, grad] = getGradAnalytical(G_f, dG_f, x);
% Z_mg = reshape(val,size(X_mg));
% contourf(X_mg,Y_mg,Z_mg,100,'LineStyle','none')
% hold on
% contour(X_mg,Y_mg,Z_mg,[0,0.001],'LineStyle','-','LineColor','k','LineWidth',2)
% l = streamslice(X_mg,Y_mg,U_nominal,V_nominal);

% plot optimized trajectory
% plot(x_opt(1,:), x_opt(2,:),'LineWidth',2,'Color','g');
% plot(x_opt(1,:), x_opt(2,:),'g*');
% plot(x_opt(1,1), x_opt(2,1),'r*');
% plot(attr(1), attr(2),'g*');
% xlim([-10 10]);
% ylim([-10 10])
figure(2)
title('Control input')
hold on
stairs(u_opt(:,1),'LineWidth',2);
%stairs(u_opt(:,2),'LineWidth',2);
legend('u1','u2')

%% animation plot
f_anim = figure(3);
figure(f_anim)
ax_anim = axes(f_anim);
axes(ax_anim)
ax_anim.XLim = [-10; 10];
ax_anim.YLim = [-10; 10];
axis equal
hold on

for i = 1:1:size(x_opt,2)-1
    cla(ax_anim);
    l = streamslice(ax_anim, X_mg,Y_mg,U_nominal,V_nominal);
    circle([obs_x;obs_y],obs_r,'LineWidth',2);
    plot(ax_anim, x_opt(1, 1:i), x_opt(2, 1:i),'r-','LineWidth',2);
    plot(ax_anim, x_opt(1, 1:i), x_opt(2, 1:i),'r*');
    plot(ax_anim, all_traj(:, 1, i), all_traj(:, 2, i),'g--','LineWidth',2);
    drawnow
end
%% some functions
function [t0, x0, u0] = shift(dt, t0, x0, u,f)
    st = x0;
    con = u(1,:)';
    f_value = f(st,con);
    st = st + (dt*f_value);
    x0 = full(st);
    
    t0 = t0 + dt;
    u0 = [u(2:size(u,1),:);u(size(u,1),:)];
end
