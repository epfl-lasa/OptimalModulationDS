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
x1_max = q_max(1); x1_min = q_min(1);
x2_max = q_max(2); x2_min = q_min(2);

A = [-5 0.2; 0.6 -1];
%A = -eye(2);
attr = [-0.8*pi; -0.5*pi];
attr = [-1.25; -1.1];
init_pos = [2; 2];

rhs = @(state,attr, k) k*A*(state - attr)/max(0.5,norm(state - attr));
obs_pos = [9e7; 0];
obs_r = 0;
thr = 0;
dt = 0.01;

%% THE SIMULATION LOOP STARTS HERE
%-------------------------------------------
x0 = init_pos;
%x0 = [0.8*pi ; 0];    % initial condition.
x_ref = attr;% Reference posture.
x_opt(:,1) = x0; % xx contains the history of states
mpciter = 0;
main_loop = tic;
while(norm(x0-x_ref)>1e-2)
    % Apply the control and shift the solution
    %use RK
    u0 = 0;
    inp = [x0; obs_pos];
    pos_inp = [inp; sin(inp); cos(inp)];
    dst = y_f(pos_inp)-obs_r;
    dst = 1;
    x0 = x0+dt*rhs(x0, attr, min(1,dst));
    x_opt(:,mpciter+2) = x0;
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
