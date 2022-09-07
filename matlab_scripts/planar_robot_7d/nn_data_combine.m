clc
clear all
ds = [];
for i = 1:1:8
    load(['data/100k',num2str(i),'.mat'])
    ds = [ds; dataset];
end
dataset = ds;
save('data/dataset_nn.mat','dataset')
