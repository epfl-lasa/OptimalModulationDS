clc
clear all force
close all force
pos_enc = @(x)[x sin(x) cos(x)];
%pos_enc = @(x)[x];

load('data/dataset_nn.mat')
dataset = dataset(randperm(length(dataset)),:);

% only keep point where ee is close to the goal
tt_ratio = 0.98;
idx_tt = floor(tt_ratio*length(dataset));
data_train = dataset(1:idx_tt,:);
data_test = dataset((idx_tt+1):end,:);
sz = 128;
layers = [
    sequenceInputLayer((size(data_train,2)-1)*3)
    %sequenceInputLayer((size(data_train,2)-1))
    fullyConnectedLayer(sz)
    tanhLayer()
    fullyConnectedLayer(sz)
    tanhLayer()
    fullyConnectedLayer(sz)
    tanhLayer()
    fullyConnectedLayer(sz)
    tanhLayer()
    fullyConnectedLayer(1)
    regressionLayer()];
X = data_train(:,2:end);
X = [X sin(X) cos(X)];
Y = data_train(:,1)-0.5;
options = trainingOptions('rmsprop', ...
    'MaxEpochs',100000,...
    'InitialLearnRate',1e-3, ...
    'Verbose',true, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',10000, ...
    'Plots','training-progress');

    %'Plots','training-progress',...
    %'ValidationData',{data_test(:,2:end)',data_test(:,1)'});

%%
%net1 = feedforwardnet([100,100,100,100],'trainlm');
% net1.inputs{1}.processFcns = {};
% net1.outputs{3}.processFcns = {};
%net1 = train(net1,X',Y','useGPU','yes');
%y = net1(x);
%%
net = trainNetwork(X',Y',layers,options);
save('data/net128_pos','net')
pred_res = predict(net, pos_enc(data_test(:,2:end))'); 
ground_truth = data_test(:,1);
err = (pred_res'-ground_truth);
hist(err);
mean(err);
% net = assembleNetwork(layers);
% tic
% pred_res = predict(net, data_test');
% toc

%%
% q = [0,0,2,3];
% predict(net, pos_enc(q)')
%predict(net, q')
%%
% clc
% clear all force
% close all force
% pos_enc = @(x)[x sin(x) cos(x)];
% %pos_enc = @(x)[x];
% 
% load('data/dataset_nn.mat')
% dataset = dataset(randperm(length(dataset)),:);
% 
% % only keep point where ee is close to the goal
% tt_ratio = 0.9;
% idx_tt = floor(tt_ratio*length(dataset));
% data_train = dataset(1:idx_tt,:);
% data_test = dataset((idx_tt+1):end,:);
% load('data/net50.mat')
% pred_res = predict(net, pos_enc(data_test(:,2:end))'); 
% ground_truth = data_test(:,1);
% err = (pred_res'-ground_truth);
% s = 20;
% [pred_res(1:s)' ground_truth(1:s) err(1:s)]