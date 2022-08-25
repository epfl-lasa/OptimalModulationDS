clc
clear all
close all force
pos_enc = @(x)[x sin(x) cos(x)];
r = [4 4 0];
d = [0 0 0];
alpha = [0 0 0];
base = eye(4);
%%
load('data/200k_nn.mat')
%dataset = dataset(randperm(length(dataset)),:);
jpos = dataset(:,2:3);
pts = dataset(:,4:5);
dst = dataset(:,1);
jpos_uniq = unique(jpos,'rows');
u_idx = 1;
idx = find(abs(sum(jpos-jpos_uniq(u_idx,:),2))<1e-5);
%plot(pts(idx,1),pts(idx,2),'b.')
%axis equal

x = pts(idx,1);
y = pts(idx,2);
z = dst(idx)-1;
n = 100;
[X, Y] = meshgrid(linspace(min(x),max(x),n), linspace(min(y),max(y),n));
Z = griddata(x,y,z,X,Y);
[~, ctr] = contourf(X,Y,Z,100,'LineStyle','none');
hold on
lvl = 0;
%[~, cr1] = contour(X,Y,Z,[lvl,lvl+0.001],'LineStyle','-','LineColor','k','LineWidth',2);
plot(pts(idx,1),pts(idx,2),'b.')
create_r(gca, jpos_uniq(u_idx,:), r, d , alpha, base);

%%
f2 = figure(2);
ax_h = axes(f2,'View',[0 90]);
axis equal
hold on
ax_h.XLim = [-10 10];
ax_h.YLim = [-10 10];

for i = 1:1:length(jpos_uniq)
    create_r(ax_h, jpos_uniq(i,:), r, d , alpha, base);
end

function handle = create_r(ax_h,j_state,r,d,alpha,base)
    pts = calc_fk(j_state,r,d,alpha,base);
    handle = plot(ax_h,pts(:,1),pts(:,2),'LineWidth',2,...
        'Marker','o','MarkerFaceColor','k','MarkerSize',4);
end

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
