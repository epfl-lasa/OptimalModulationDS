clc
clear all
close all
%% nominal ds
x0 = [-9 0]';
%specify attractor
goal_pos = [8 0]';
%system dynamics
A = 1*[-1 0; 0 -1];
f = @(x, goal_pos)A*(x-goal_pos);
f = @(x, goal_pos)A*(x-goal_pos)/max(norm(x-goal_pos),1e-3);

% plotting
figure(1)
hold on
x_span=linspace(-10,10,50); 
y_span=linspace(-10,10,50); 
[X_mg,Y_mg] = meshgrid(x_span, y_span);
x=[X_mg(:) Y_mg(:)]';
x_dot = rotMat(0)*f(x, goal_pos);
U_nominal = reshape(x_dot(1,:), length(y_span), length(x_span));
V_nominal = reshape(x_dot(2,:), length(y_span), length(x_span));
l = streamslice(X_mg,Y_mg,U_nominal,V_nominal);
axis equal
title('Nominal DS')

%% obstacle
load('obstacles/net_bean.mat')
[y_f, dy_f] = tanhNN(net);
[val, grad] = getGradAnalytical(y_f, dy_f, x);
Z_mg = reshape(val,size(X_mg));
contourf(X_mg,Y_mg,Z_mg,100,'LineStyle','none')
hold on
contour(X_mg,Y_mg,Z_mg,[0,0.001],'LineStyle','-','LineColor','k','LineWidth',2)
l = streamslice(X_mg,Y_mg,U_nominal,V_nominal);

%%
% Create non-linear MPN object with two states x, input u and output y
nu = 1; nx = 2; ny = 2;
nlobj = nlmpc(nx, ny, nu);
nlobj.Model.NumberOfParameters = 2;

% Define sampling (Ts [s]), control and prediction (Tf) time [s]
Ts = 1;
nlobj.Model.IsContinuousTime = true;
nlobj.Ts = Ts;
nlobj.PredictionHorizon = 25;
nlobj.ControlHorizon = 25;

%define model dynamics
nlobj.Model.StateFcn = @(x, u, goalPos, y_f)  rotMat(u) * f(x, goalPos);
%nlobj.Model.StateFcn = @myStateFcn;


nlobj.Model.OutputFcn = @(x, u, goalPos, y_f) goalPos;


% Define state and input limits
for i = 1:nx
    nlobj.States(i).Min = -10; 
    nlobj.States(i).Max = 10; 
end

for i = 1:1:nu
    nlobj.MV(i).Min = -2*pi;
    nlobj.MV(i).Max = 2*pi;
    nlobj.MV(i).RateMin = -1;
    nlobj.MV(i).RateMax = 1;
end
% Tune solver parameters for better performances
nlobj.Optimization.SolverOptions.ConstraintTolerance = 1e-2;
nlobj.Optimization.SolverOptions.MaxIterations = 50;

%custom cost fcn (minimize rotation)
nlobj.Optimization.CustomCostFcn = @minControl;

%custom eq constraint (goal reaching on the horizon)
nlobj.Optimization.CustomEqConFcn = @myEqConFunction;

%custom eq constraint (collision avoidance)
nlobj.Optimization.CustomIneqConFcn = @myIneqConFunction;

nloptions = nlmpcmoveopt;

nloptions.Parameters = {goal_pos, y_f};

u0 = 0;
validateFcns(nlobj, x0, u0, [], {goal_pos, y_f});
tic
[mv, opt, optimalSolution] = nlmpcmove(nlobj, x0, u0, [], [], nloptions); 
toc
%% plotting
traj = optimalSolution.Xopt;
optU = optimalSolution.MVopt;
xd = traj*0;
xd(1,:) = x0;
for i = 1:1:nlobj.PredictionHorizon+1
    xd(1+i,:) = (xd(i,:)'+Ts*rotMat(optU(i))*f(xd(i,:)', goal_pos))';
end
plot(traj(:,1),traj(:,2),'r','LineWidth',2)
plot(traj(:,1),traj(:,2),'r*','LineWidth',1)

plot(x0(1),x0(2),'b*')
plot(goal_pos(1),goal_pos(2),'g*')

axis equal
xlim([-10 10])
ylim([-10 10])
figure(2)
hold on
plot(optU, 'LineWidth', 2)
xlabel('Timestep')
ylabel('\theta')
title('Modulation rotation angle')
%% costs and constrains

function ceq = myEqConFunction(X, U, data, goalPos, y_f)
    % This function enforces the constraint to reach a specific target at the
    % end of the horizon
    ceq = norm(X(end,:) - goalPos');
    %ceq = [ceq;vecnorm(U(10:end),2,2)];
end

function cineq = myIneqConFunction(X, U, e, data, goalPos, y_f)
    % collision avoidance (Gamma<0)
    cineq = -1*double(y_f(X')-2)';
    y_f(X')
end


function cost = minControl(X, U, e, data, goalPos, y_f)
%     dt = data.PredictionHorizon;
%     cost = 0;
%     for i = 1:data.PredictionHorizon
        %cost = cost + norm(U(i, :));
        %eyedif = rotMat(U(i)) - eye(2);
        %cost = cost+norm(eyedif);
%     end
%     cost = sum(abs(U))+5*sum(abs(diff(U)));
    cost = sum(abs(U));
   % cost = sum(abs(diff(U(1:20))));

    %X(end,:)
end

function rotMat = rotMat(u)
    rotMat = [cos(u) -sin(u); sin(u) cos(u)];
end