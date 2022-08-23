clc
clear all
close all

%% generate data
x_span=linspace(-10,10,100); 
y_span=linspace(-10,10,100); 
[X_mg,Y_mg] = meshgrid(x_span, y_span);
x=[X_mg(:) Y_mg(:)]';
d1 = sqrt(x(1,:).^2+(x(2,:)+1).^2)-1.75;
d2 = sqrt(x(1,:).^2+(x(2,:)-1).^2)-1.75;
d3 = sqrt((x(1,:)+2).^2+(x(2,:)+3).^2)-1.75;
d4 = sqrt((x(1,:)+2).^2+(x(2,:)-3).^2)-1.75;
d = min([d1;d2;d3;d4],[],1);
% Z_mg = reshape(d,size(X_mg));
% contourf(X_mg,Y_mg,Z_mg,100,'LineStyle','none')
% hold on
% contour(X_mg,Y_mg,Z_mg,[0,0.001],'LineStyle','-','LineColor','k','LineWidth',2)
% axis equal

%% train MLP
% net1 = feedforwardnet([10,10],'trainlm');
% net1.inputs{1}.processFcns = {};
% net1.outputs{3}.processFcns = {};
% net1 = train(net1,x,d);
% y = net1(x);

options = trainingOptions('rmsprop','MaxEpochs',5000,...
    'ExecutionEnvironment','cpu',...
    'InitialLearnRate',1e-2, 'Verbose',true, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',1000, ...
    'Plots','training-progress', 'MiniBatchSize',250000);
sz = 15;
layers = [
    featureInputLayer(2)
    fullyConnectedLayer(sz) %2 x sz
    tanhLayer()
    fullyConnectedLayer(sz) %sz x sz
    tanhLayer()
    fullyConnectedLayer(sz) %sz x sz
    tanhLayer()
    fullyConnectedLayer(1) %sz x 1
    regressionLayer()];
%tic
net = trainNetwork(x',d',layers,options);
save('net_bean.mat','net')
%toc
%tic
%y = predict(net, x')';
%toc

%% autodiff way (SLOW!)
% load('net_circle.mat')
% lgraph = layerGraph(net);
% lgraph = removeLayers(lgraph,["regressionoutput"]);
% dlnet = dlnetwork(lgraph);
% x_arr = dlarray(x,'CB');
% tic
% [val, grad] = getGrad(dlnet, x_arr);
% toc
%% analytical tanh way (fast!)
%load('net_bean.mat')
load('net_circle.mat')

[y_f, dy_f] = tanhNN(net);
tic
[val, grad] = getGradAnalytical(y_f, dy_f, x);
toc
%% plotting
figure()
Z_mg = reshape(val,size(X_mg));
contourf(X_mg,Y_mg,Z_mg,100,'LineStyle','none')
hold on
contour(X_mg,Y_mg,Z_mg,[0,0.001],'LineStyle','-','LineColor','k','LineWidth',2)
axis equal
streamslice(X_mg,Y_mg,reshape(grad(1,:),size(X_mg)),reshape(grad(2,:),size(Y_mg)));



