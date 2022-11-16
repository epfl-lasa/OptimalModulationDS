clc
clear all
load ionosphere
rng(1); % For reproducibility
N = 1000;
D = 10;
X = rand(N,D);
Y = round(rand(N,1));
X(Y==1,:) = X(Y==1,:)+10;
tic
SVMModel = fitcsvm(X,Y,'Standardize',false,'KernelFunction','RBF',...
    'KernelScale','auto');
toc