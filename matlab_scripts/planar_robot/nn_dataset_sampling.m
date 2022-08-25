%% sampling for nn
clc
close all
clear all

%% parameters definition
l1=4; l2=4; %shoulder length
%dh elements
r = [l1 l2 0];
d = [0 0 0];
alpha = [0 0 0];
base = eye(4);
k1 = 1.1; k2 = 1.1;
q_min = [-k1*pi, -k2*pi, -11,-11];
q_max = [k1*pi, k2*pi, 11, 11];

%% sampling procedure
N = 200000;
%r_vec = rand(N, size(q_min,2));
%data = q_min+r_vec.*(q_max-q_min);
%data = data(:,[1,2]);
tic
dataset = zeros(N,5);
x_span = linspace(q_min(3),q_max(3),10);
y_span = linspace(q_min(4),q_max(4),10);
[X_mg,Y_mg] = meshgrid(x_span, y_span);
x = [X_mg(:) Y_mg(:)];
%% fill up states
ns = length(x);
ds_close = [];
n_sampl_links = 4;
n_jpos = floor(N/ns)+1;
n_jpos = 1000;
i_ds = 1;
for i = 1:1:n_jpos
    jpos = q_min(1:2)+rand(1,2).*(q_max(1:2)-q_min(1:2));
    %first generate grid points
    for j = 1:1:ns
        dataset(i_ds, 2:3) = jpos;
        dataset(i_ds, 4:5) = x(j,:)+0.5*(2*rand(1,2)-1);
        i_ds = i_ds + 1;
    end

    %add points around links
    rob_pos = calc_fk(jpos,r,d,alpha,base);
    full_rob_pos = robot_ptcloud(rob_pos,n_sampl_links);
    for j = 2:1:length(full_rob_pos)
        %dataset(i_ds, 2:3) = jpos;
        %dataset(i_ds, 4:5) = full_rob_pos(j,1:2);
        %i_ds = i_ds + 1;
        for k = 1:1:20
            dataset(i_ds, 2:3) = jpos;
            dataset(i_ds, 4:5) = full_rob_pos(j,1:2)+2*(2*rand(1,2)-1);
            i_ds = i_ds + 1;
        end
    end
    i_ds
end
toc
disp('Samples generated!')

%%
n_sampl_links = 100;
for i = 1:1:length(dataset)
    state = dataset(i,2:end);
    %state = q_min+rand(1,4).*(q_max-q_min);
    rob_pos = calc_fk(state(1:2),r,d,alpha,base);
    full_rob_pos = robot_ptcloud(rob_pos,n_sampl_links);
    dst = min(vecnorm(full_rob_pos(:,1:2) - state(3:4),2,2)); %get mindist
    dataset(i,1)=dst;
%     if dst <1
%         break
%     end
end
% dataset(i,:)
% plot(full_rob_pos(:,1),full_rob_pos(:,2),'b.')
% hold on
% plot(state(3), state(4), 'r*')
% axis equal
toc
save('data/200k_nn.mat','dataset')

function pts = calc_fk(j_state,r,d,alpha,base)
    P = dh_fk(j_state,r,d,alpha,base);
    pts = zeros(3,3);
    for i = 1:1:3
        v = [0,0,0];
        R = P{i}(1:3,1:3);
        T = P{i}(1:3,4);
        p = v*R'+T';
        pts(i,:) = p;
    end
end

function pts = robot_ptcloud(rob_pos, N)
    n_links = size(rob_pos,1)-1;
    vecspace = @(v1,v2,k) v1+linspace(0,1,k)'.*(v2-v1);
    pts = zeros(n_links*N,size(rob_pos,2));
    for i = 1:1:n_links
        pts((i-1)*N+1:i*N, :) = vecspace(rob_pos(i,:),rob_pos(i+1,:), N);
    end
end
