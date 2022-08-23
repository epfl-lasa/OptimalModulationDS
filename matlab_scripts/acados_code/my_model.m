function model = my_model(attr, A, G_f)

import casadi.*

%% system dimensions
nx = 2;
nu = 2;

%% system parameters
%attr = [8;0];
model.attr = attr;
model.A = A;
%% symbolic variables
sym_x = SX.sym('x', nx, 1);
sym_xdot = SX.sym('xdot', nx, 1);
sym_u = SX.sym('u',nu,1);

%% dynamics
sin_t = sin(sym_u(1));
cos_t = cos(sym_u(1));
dist = norm(sym_x - attr);
%A = [-1 0; 0 -1];
R = [cos_t -sin_t; sin_t cos_t];
f = R * A * (sym_x - attr);
%f = sym_u + (sym_x - attr)/max(dist,0.1);
f = -A*(sym_x - attr) + sym_u;

expr_f_expl = f;
expr_f_impl = sym_xdot - expr_f_expl;

%% constraints
%expr_h = sym_u;
G_val = G_f([sym_x(1);sym_x(2)]);
expr_h = G_val;
%expr_h = sqrt(sym_x' * sym_x)+2;
%expr_h = 0;
%expr_h = sym_u;
%% cost
W_x = diag([1, 1]);
W_u = 1;
expr_ext_cost_e = (sym_x-attr)'* W_x * (sym_x-attr); %reach goal
%expr_ext_cost_e = (sym_x)'* W_x * (sym_x); % reach goal
expr_ext_cost = expr_ext_cost_e + sym_u' * W_u * sym_u; %minimize u & reach goal


%% populate structure
model.nx = nx;
model.nu = nu;
model.sym_x = sym_x;
model.sym_xdot = sym_xdot;
model.sym_u = sym_u;
model.expr_f_expl = expr_f_expl;
model.expr_f_impl = expr_f_impl;
model.expr_h = expr_h;
model.expr_ext_cost = expr_ext_cost;
model.expr_ext_cost_e = expr_ext_cost_e;

end
