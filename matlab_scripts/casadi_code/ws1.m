clc
clear all
close all
addpath('/home/michael/Documents/EPFL/Projects/casadi-linux-matlabR2014b-v3.5.5')
import casadi.*

x = SX.sym('w');
obj = x^2-6*x+13;

g = [];
P = [];

OPT_variables = x;
nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g,'p', P);

opts = struct;
opts.ipopt.max_iter = 100;
opts.ipopt.print_level = 0;
opts.print_time = 0;
opts.ipopt.acceptable_tol = 1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-8;
solver = nlpsol('solver', 'ipopt', nlp_prob, opts);

args = struct;
args.lbx = -inf;
args.ubx = inf;
args.lbg = -inf;
args.ubg = inf;
args.p = [];
args.x0 = -0.5;

sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
    'lbg', args.lbg, 'ubg', args.ubg, 'p', args.p);
x_sol = full(sol.x);
min_value = full(sol.f);


















