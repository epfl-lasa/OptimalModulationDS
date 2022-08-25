clc
clear all
close all
%% 1. linear 2d DS
%specify attractor
x_goal = [8 0]';
%system dynamics
A = [-1 0; 0 -1];
f = @(x)A*(x-x_goal);
%modulation matrix
M = @(x)[1 0; 0 1];

% plotting
figure(1)
x_span=linspace(-10,10,50); 
y_span=linspace(-10,10,50); 
[X_mg,Y_mg] = meshgrid(x_span, y_span);
x=[X_mg(:) Y_mg(:)]';
x_dot = M(x)*f(x);
U_nominal = reshape(x_dot(1,:), length(y_span), length(x_span));
V_nominal = reshape(x_dot(2,:), length(y_span), length(x_span));
l = streamslice(X_mg,Y_mg,U_nominal,V_nominal);
axis tight
title('Nominal DS')
%% obstacle
load('obstacles/net_bean.mat')
[y_f, dy_f] = tanhNN(net);
[val, grad] = getGradAnalytical(y_f, dy_f, x);
%% plotting
figure(2)

Z_mg = reshape(val,size(X_mg));
contourf(X_mg,Y_mg,Z_mg,100,'LineStyle','none')
hold on
contour(X_mg,Y_mg,Z_mg,[0,0.001],'LineStyle','-','LineColor','k','LineWidth',2)
axis equal
U_grad = reshape(grad(1,:),size(X_mg));
V_grad = reshape(grad(2,:),size(Y_mg));
streamslice(X_mg,Y_mg, U_grad, V_grad);
%streamslice(X_mg,Y_mg, -V_grad, U_grad);

title('Obstacle repuslion field')
% starty = -3:1:3;
% startx = zeros(size(starty));
% lines = streamline(X_mg,Y_mg,-V_grad+0.5*U_grad,U_grad+0.5*V_grad,startx,starty);
% set(lines, 'LineWidth',1.5, 'Color','r')

%%

t_v = @(v)[v(2) -v(1)]';
mm = @(x,rep_vec, l1,l2) [rep_vec t_v(rep_vec)] * [l1 0 ;0 l2] * [rep_vec t_v(rep_vec)]';
x_dot_mod = x_dot*0;
for i = 1:1:length(x)
    dist = max(y_f(x(:,i)),1e-8)+0.5;
    rep_vec = dy_f(x(:,i))';
    rep_vec = rep_vec/norm(rep_vec);
    l1 = 1-1/(dist);
    l2 = 1+1/(dist);
%     l1 = 1;
%     l2 = 1;
    x_dot_mod(:,i) = mm(x(:,i), rep_vec, l1,l2) * f(x(:,i));
end
U_mod = reshape(x_dot_mod(1,:), length(y_span), length(x_span));
V_mod = reshape(x_dot_mod(2,:), length(y_span), length(x_span));
%close all
figure()
l = streamslice(X_mg,Y_mg,U_mod,V_mod);
axis equal
title('Modulated DS')
hold on
starty = -10:1:10;
startx = zeros(size(starty))-9;
lines = streamline(X_mg,Y_mg,U_mod,V_mod,startx,starty);
set(lines, 'LineWidth',1.5, 'Color','r')
contour(X_mg,Y_mg,Z_mg,[0,0.001],'LineStyle','-','LineColor','k','LineWidth',2)

