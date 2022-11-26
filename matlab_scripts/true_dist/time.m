clc
clear all
% euler integration 
x0 = 0;
u = 0.01;
dt = 0.01;
xcur = x0;
tic;
for j = 1:1:100
    for i = 1:1:10000
        xcur = xcur + u*dt;
    end
end
tf = toc;
tf/j