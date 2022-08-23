%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise Script for  Chapter 1 of:                                      %
% "Robots that can learn and adapt" by Billard, Mirrazavi and Figueroa.   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2020 Learning Algorithms and Systems Laboratory,          %
% EPFL, Switzerland                                                       %
% Author:  Alberic de Lajarte                                             %
% email:   alberic.lajarte@epfl.ch                                        %
% website: http://lasa.epfl.ch                                            %
%                                                                         %
% Permission is granted to copy, distribute, and/or modify this program   %
% under the terms of the GNU General Public License, version 2 or any     %
% later version published by the Free Software Foundation.                %
%                                                                         %
% This program is distributed in the hope that it will be useful, but     %
% WITHOUT ANY WARRANTY; without even the implied warranty of              %
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General%
% Public License for more details                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Question 2: Compute optimal trajectory  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;
filepath = fileparts(which('chp1_ex1_2_solution.m'));
addpath(genpath(fullfile(filepath, '..', '..', 'libraries', 'book-robot-simulation')));

% Create robot from the custom RobotisWrapper class
robot = RobotisWrapper();

optimal_control = MPC4DOF(robot);
target_position = [0.1; -0.3; 0.1];

%% Task 1: Minimum time
start = tic;
optimal_control.nlSolver.Optimization.CustomCostFcn = @minimumTime;
optimalSolution = optimal_control.solveOptimalTrajectory(target_position);
toc(start);
optimal_control.showResults(optimalSolution, target_position, 'Minimal Time');
disp("Press space to continue..."); pause();

%% Task 2: Minimum Cartesian distance
start = tic; 
optimal_control.nlSolver.Optimization.CustomCostFcn = @minimumTaskDistance;
optimalSolution = optimal_control.solveOptimalTrajectory(target_position);
toc(start);
optimal_control.showResults(optimalSolution, target_position, 'Minimal Task Distance');
disp("Press space to continue..."); pause();

%% Task 3: Minimum joint distance 
start = tic; 
optimal_control.nlSolver.Optimization.CustomCostFcn = @minimumJointDistance;
optimalSolution = optimal_control.solveOptimalTrajectory(target_position);
toc(start);
optimal_control.showResults(optimalSolution, target_position, 'Minimal Joint Distance');

%% %%%%%%%%%%%%% User defined cost functions %%%%%%%%%%%%% %%

% The robot arm is modeled with the following state space representation:
% - X: state vector = 4 joint angles of the robot arm
% - U(1:4): input vector = 4 joint speed
% - U(5): constant parameter = final time of the horizon, at which the robot
% arm should reach a specified target

%% OVERWRITE COST VARIABLE IN EACH FUNCTION %%

% Task 1: Minimum time 
% This function integrates the time scaling parameter u(5) to minimize
% trajectory time
function cost = minimumTime(X, U, e, data, robot, target)

    cost = U(1, 5);
end


% Task 2: Minimum distance in task space 
% This function integrates dx = J*dq to minimize Cartesian trajectory length
% You can obtain the Jacobian J at configuration q using
% J = robot.fastJacobian(q)
% USE THE SQUARE OF THE NORM FOR NUMERICAL STABILITY
function cost = minimumTaskDistance(X, U, e, data, robot, target)

    dt = U(1, 5) / data.PredictionHorizon;
    cost = 0;
    for i = 1:data.PredictionHorizon
        cost = cost + sum((robot.fastJacobian(X(i,:)) * U(i, 1:4)' * dt) .^2);
    end
end


% Task 3: Minimum distance in joint space 
% This function integrates dq = u(1:4)*dt to minimize joint trajectory length
% USE THE SQUARE OF THE NORM FOR NUMERICAL STABILITY
function cost = minimumJointDistance(X, U, e, data, robot, target)

    dt = U(1, 5) / data.PredictionHorizon;
    cost = 0;
    for i = 1:data.PredictionHorizon
        cost = cost + sum((U(i, 1:4) * dt).^2);

        % Alternative solution: minimize joint difference between two steps 
        %cost = cost + sum((X(i+1,:) - X(i, :)).^2); 
    end
end

