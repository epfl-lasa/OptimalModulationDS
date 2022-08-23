%% test of native matlab interface
clc
clear all
close all
addpath('..')
% check that env.sh has been run
env_run = getenv('ENV_RUN');
if (~strcmp(env_run, 'true'))
	error('env.sh has not been sourced! Before executing this example, run: source env.sh');
end
load('../obstacles/net_circle.mat')
[G_f, dG_f] = tanhNN(net);

%% arguments
compile_interface = 'auto'; %'auto';
codgen_model = 'true';
gnsf_detect_struct = 'true';

% discretization
N = 50;
h = 0.5;

nlp_solver = 'sqp';
%nlp_solver = 'sqp_rti';
%nlp_solver_exact_hessian = 'false';
nlp_solver_exact_hessian = 'true';
regularize_method = 'no_regularize';
%regularize_method = 'project';
%regularize_method = 'project_reduc_hess';
%regularize_method = 'mirror';
%regularize_method = 'convexify';
nlp_solver_max_iter = 100; %100;
nlp_solver_tol_stat = 1e-8;
nlp_solver_tol_eq   = 1e-8;
nlp_solver_tol_ineq = 1e-8;
nlp_solver_tol_comp = 1e-8;
nlp_solver_ext_qp_res = 0;
%qp_solver = 'partial_condensing_hpipm';
qp_solver = 'full_condensing_hpipm';
qp_solver_cond_N = 1;
qp_solver_cond_ric_alg = 0;
qp_solver_ric_alg = 0;
qp_solver_warm_start = 0;
qp_solver_max_iter = 100;
sim_method = 'erk';
%sim_method = 'irk';
%sim_method = 'irk_gnsf';
sim_method_num_stages = 4;
sim_method_num_steps = 1;
%cost_type = 'linear_ls';
cost_type = 'ext_cost';
model_name = 'my_model';


%% create model entries
attr = [8 0]';
A = [-1  0;
      0 -1];

model = my_model(attr, A, G_f);
x0 = [-8; 0];

% dims
T = N*h; % horizon length time
nx = model.nx;
nu = model.nu;
ny = 1; % number of outputs in lagrange term (why?)
ny_e = nx; % number of outputs in mayer term (same as state)

% constraints
Jbx = eye(nx); % all states bounded
lbx = -10*ones(nx, 1);
ubx =  10*ones(nx, 1);
Jbu = eye(nu); % all controls bounded
lbu = -2*pi*ones(nu, 1);
ubu =  2*pi*ones(nu, 1);


%% acados ocp model
ocp_model = acados_ocp_model();
ocp_model.set('name', model_name);
ocp_model.set('T', T);

% symbolics
ocp_model.set('sym_x', model.sym_x);
if isfield(model, 'sym_u')
	ocp_model.set('sym_u', model.sym_u);
end
if isfield(model, 'sym_xdot')
	ocp_model.set('sym_xdot', model.sym_xdot);
end
% cost
ocp_model.set('cost_type', cost_type);
ocp_model.set('cost_type_e', cost_type);
ocp_model.set('cost_expr_ext_cost', model.expr_ext_cost);
ocp_model.set('cost_expr_ext_cost_e', model.expr_ext_cost_e);

% dynamics
if (strcmp(sim_method, 'erk'))
	ocp_model.set('dyn_type', 'explicit');
	ocp_model.set('dyn_expr_f', model.expr_f_expl);
else % irk irk_gnsf
	ocp_model.set('dyn_type', 'implicit');
	ocp_model.set('dyn_expr_f', model.expr_f_impl);
end
% constraints
ocp_model.set('constr_x0', x0);
ocp_model.set('constr_Jbx', Jbx);
ocp_model.set('constr_lbx', lbx);
ocp_model.set('constr_ubx', ubx);
ocp_model.set('constr_Jbu', Jbu);
ocp_model.set('constr_lbu', lbu);
ocp_model.set('constr_ubu', ubu);

%collision cost
ocp_model.set('constr_expr_h', model.expr_h)
ocp_model.set('constr_lh', 1);
ocp_model.set('constr_uh', 10000);

%disp('ocp_model.model_struct')
%disp(ocp_model.model_struct)


%% acados ocp opts
ocp_opts = acados_ocp_opts();
ocp_opts.set('compile_interface', compile_interface);
ocp_opts.set('codgen_model', codgen_model);
ocp_opts.set('param_scheme_N', N);
ocp_opts.set('nlp_solver', nlp_solver);
ocp_opts.set('nlp_solver_exact_hessian', nlp_solver_exact_hessian);
ocp_opts.set('regularize_method', regularize_method);
ocp_opts.set('nlp_solver_ext_qp_res', nlp_solver_ext_qp_res);
if (strcmp(nlp_solver, 'sqp'))
	ocp_opts.set('nlp_solver_max_iter', nlp_solver_max_iter);
	ocp_opts.set('nlp_solver_tol_stat', nlp_solver_tol_stat);
	ocp_opts.set('nlp_solver_tol_eq'  , nlp_solver_tol_eq);
	ocp_opts.set('nlp_solver_tol_ineq', nlp_solver_tol_ineq);
	ocp_opts.set('nlp_solver_tol_comp', nlp_solver_tol_comp);
end
ocp_opts.set('qp_solver', qp_solver);
ocp_opts.set('qp_solver_cond_ric_alg', qp_solver_cond_ric_alg);
ocp_opts.set('qp_solver_warm_start', qp_solver_warm_start);
ocp_opts.set('qp_solver_iter_max', qp_solver_max_iter);
if (~isempty(strfind(qp_solver, 'partial_condensing')))
	ocp_opts.set('qp_solver_cond_N', qp_solver_cond_N);
end
if (strcmp(qp_solver, 'partial_condensing_hpipm'))
	ocp_opts.set('qp_solver_ric_alg', qp_solver_ric_alg);
end
ocp_opts.set('sim_method', sim_method);
ocp_opts.set('sim_method_num_stages', sim_method_num_stages);
ocp_opts.set('sim_method_num_steps', sim_method_num_steps);
if (strcmp(sim_method, 'irk_gnsf'))
	ocp_opts.set('gnsf_detect_struct', gnsf_detect_struct);
end

%disp('ocp_opts');
%disp(ocp_opts.opts_struct);


%% acados ocp
% create ocp
ocp_opts.set('print_level', 2);

ocp = acados_ocp(ocp_model, ocp_opts);
ocp;
%disp('ocp.C_ocp');
%disp(ocp.C_ocp);
%disp('ocp.C_ocp_ext_fun');
%disp(ocp.C_ocp_ext_fun);
%ocp.model_struct

% solve
tic;

% solve ocp
ocp.solve();

time_ext = toc;
% TODO: add getter for internal timing
fprintf(['time for ocp.solve (matlab tic-toc): ', num2str(time_ext), ' s\n'])

% get solution
u_opt = ocp.get('u');
x_opt = ocp.get('x');

%% evaluation
status = ocp.get('status');
sqp_iter = ocp.get('sqp_iter');
time_tot = ocp.get('time_tot');
time_lin = ocp.get('time_lin');
time_reg = ocp.get('time_reg');
time_qp_sol = ocp.get('time_qp_sol');

fprintf('\nstatus = %d, sqp_iter = %d, time_ext = %f [ms], time_int = %f [ms] (time_lin = %f [ms], time_qp_sol = %f [ms], time_reg = %f [ms])\n', status, sqp_iter, time_ext*1e3, time_tot*1e3, time_lin*1e3, time_qp_sol*1e3, time_reg*1e3);

%ocp.print('stat');


%% figures
figure(1)
hold on
%nominal ds
x_span=linspace(-10,10,50); 
y_span=linspace(-10,10,50); 
[X_mg,Y_mg] = meshgrid(x_span, y_span);
x=[X_mg(:) Y_mg(:)]';
x_dot = A*(x-attr);
U_nominal = reshape(x_dot(1,:), length(y_span), length(x_span));
V_nominal = reshape(x_dot(2,:), length(y_span), length(x_span));
l = streamslice(X_mg,Y_mg,U_nominal,V_nominal);
axis equal
title('Nominal DS')

% obstacle
[val, grad] = getGradAnalytical(G_f, dG_f, x);
Z_mg = reshape(val,size(X_mg));
contourf(X_mg,Y_mg,Z_mg,100,'LineStyle','none')
hold on
contour(X_mg,Y_mg,Z_mg,[0,0.001],'LineStyle','-','LineColor','k','LineWidth',2)
l = streamslice(X_mg,Y_mg,U_nominal,V_nominal);

% plot optimized trajectory
plot(x_opt(1,:), x_opt(2,:),'LineWidth',2,'Color','g');
plot(x_opt(1,:), x_opt(2,:),'g*');
plot(x_opt(1,1), x_opt(2,1),'r*');
plot(model.attr(1), model.attr(2),'g*');
xlim([-10 10]);
ylim([-10 10])
figure(2)
stairs(u_opt(1,:),'LineWidth',2);
hold on

stairs(u_opt(2,:),'LineWidth',2);

legend('u');
%x_opt(:,end)
%model.attr
%     0: ACADOS_SUCCESS,
%     1: ACADOS_FAILURE,
%     2: ACADOS_MAXITER,
%     3: ACADOS_MINSTEP,
%     4: ACADOS_QP_FAILURE,
%     5: ACADOS_READY,
