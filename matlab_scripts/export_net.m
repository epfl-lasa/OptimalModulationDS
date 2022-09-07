clc
clear all
%%
%2d
path = 'planar_robot_2d/data/';
fname = 'net50_pos_thr.mat';
%7d
path = 'planar_robot_7d/data/';
fname = 'net128_pos.mat';

load([path, fname])
W = cell(1);
b = cell(1);
k = 1;
for i = 1:1:length(net.Layers)
    l = net.Layers(i);
    try
        W{k} = l.Weights;
        b{k} = l.Bias;
        k = k+1;
    end        
end
net_parsed.W = W;
net_parsed.b = b;
save([path, 'net_parsed.mat'], 'W','b')
