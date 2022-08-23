clc
clear all
close all
%%
% Create non-linear MPN object with two states x, input u and output y
nu = 1; nx = 2; ny = 2;
nlobj = nlmpc(nx, ny, nu);
nlobj.Model.NumberOfParameters = 1;

% Define sampling (Ts [s]), control and prediction (Tf) time [s]
Ts = 0.1;
nlobj.Model.IsContinuousTime = true;

% The optimal control problem is solved with a normalized time: [0;1]
% The real time is then obtained by multiplying it with the constant time
% scaling parameter u(5) -> t: [0; u(5)]
nlobj.Ts = Ts;
nlobj.PredictionHorizon = 30;
nlobj.ControlHorizon = 30;
f = @(x, attractor) [-1 0; 0 -1] * (x-attractor);
attractor = [5;5];

goalPos = [5;0];
x0 = [-7;0];

% Define system dynamics x_dot = f(x, u) = q_dot = u(1:4)
nlobj.Model.StateFcn = @(x, u, goalPos)  rotMat(u) * f(x, attractor);
nlobj.Model.OutputFcn = @(x, u, goalPos) goalPos;
%nlobj.Jacobian.StateFcn = @myStateJacobian;


% Define state and input limits
for i = 1:nx
    % Limit joint range to avoid singularities
    nlobj.States(i).Min = -10; 
    nlobj.States(i).Max = 10; 
end

for i = 1:1:nu
    %Limit joint speed and velocities
%     nlobj.MV(i).Min = -pi;
%     nlobj.MV(i).Max = pi;
    nlobj.MV(i).RateMin = -0.1;
    nlobj.MV(i).RateMax = 0.1;

%     %Soften rate of input change constraints
%     nlobj.MV(i).RateMinECR = 1.0;
%     nlobj.MV(i).RateMaxECR = 1.0;
end
% Tune solver parameters for better performances
nlobj.Optimization.SolverOptions.ConstraintTolerance = 1e-5;
nlobj.Optimization.SolverOptions.MaxIterations = 100;

%custom cost fcn
nlobj.Optimization.CustomCostFcn = @minControl;

%custom eq constraint
nlobj.Optimization.CustomEqConFcn = @myEqConFunction;

nloptions = nlmpcmoveopt;

nloptions.Parameters = {goalPos};

u0 = 0;
validateFcns(nlobj, x0, u0, [], {goalPos});
tic
[mv, opt, optimalSolution] = nlmpcmove(nlobj, x0, u0, [], [], nloptions); 
toc
%% plotting
traj = optimalSolution.Xopt;
optU = optimalSolution.MVopt;

figure(1)
x_span=linspace(-10,10,50); 
y_span=linspace(-10,10,50); 
[X_mg,Y_mg] = meshgrid(x_span, y_span);
x=[X_mg(:) Y_mg(:)]';
x_dot = f(x, attractor);
U_nominal = reshape(x_dot(1,:), length(y_span), length(x_span));
V_nominal = reshape(x_dot(2,:), length(y_span), length(x_span));
l = streamslice(X_mg,Y_mg,U_nominal,V_nominal);
title('Nominal DS')
hold on

plot(traj(:,1),traj(:,2),'r','LineWidth',2)
plot(x0(1),x0(2),'b*')
plot(goalPos(1),goalPos(2),'g*')

axis equal
xlim([-10 10])
ylim([-10 10])
figure(2)
hold on
plot(optU)

function ceq = myEqConFunction(X, U, data, goalPos)
    % This function enforces the constraint to reach a specific target at the
    % end of the horizon
    ceq = norm(X(end,:) - goalPos');
end


function cost = minControl(X, U, e, data, goalPos)
%     dt = data.PredictionHorizon;
    cost = 0;
    for i = 1:data.PredictionHorizon
        %cost = cost + norm(U(i, :));
        eyedif = rotMat(U(i)) - eye(2);
        cost = cost+norm(eyedif);
    end
    cost;
%     cost = norm(U);
    %X(end,:)
end

function rotMat = rotMat(u)
    rotMat = [cos(u) -sin(u); sin(u) cos(u)];
    %rotMat = [u(1) u(2); u(3) u(4)];
end