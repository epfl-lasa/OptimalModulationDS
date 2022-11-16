clc
clear all
close all force
vecspace = @(v1,v2,k) v1+linspace(0,1,k)'.*(v2-v1);

%%
r = [3 3 0];
d = r*0;
alpha = r*0;
base = eye(4);
syms j_state_sym [length(r)-1 1]
syms y_sym [3, 1] %planar => z=0
syms p_sym [3, 1]
%%
n_pts = 20;
link_sym = symbolic_fk_model(j_state_sym,r,d,alpha,base, y_sym, p_sym);
j_state = [1; 1];
y_pos = [8; 0; 0];
y_r = 1;
link_num = numeric_fk_model(j_state,r,d,alpha,base, y_pos, n_pts);

%%
%figure with robot
fig_robot = figure('Name','Two-dimensional robot');
set(fig_robot,'color','w');
ax_r = axes(fig_robot,'View',[0 90]);
axis equal
hold on
ax_r.XLim = [-10 10];
ax_r.YLim = [-10 10];
ax_r.ZLim = [-0.1 0.1];
%plot(ax_h, link_num{1}.pos(:,1), link_num{1}.pos(:,2), 'r*')
x_span = linspace(-10,10, 20);
y_span = linspace(-10,10, 20);
[X_mg,Y_mg] = meshgrid(x_span, y_span);
y=[X_mg(:) Y_mg(:) 0*Y_mg(:)]';
[dst, rep] = getClosestDistanceVec(j_state', y, link_sym, r, d, alpha, base, n_pts);
Z_mg = reshape(dst,size(X_mg));
[~, ctr_r] = contourf(ax_r, X_mg,Y_mg,Z_mg,100,'LineStyle','none');

h_rob = create_r(ax_r, link_num);

link_num = numeric_fk_model(j_state,r,d,alpha,base, y_pos, n_pts);

update_r(h_rob, link_num)

ii = link_num{1}.minidx(1);
jj = link_num{1}.minidx(2);
% rep_tmp = double(subs(link_sym{ii}.rep(jj,:), ...
%                         [j_state_sym; y_sym], ...
%                         [j_state; y_pos]));

%%
fig_jspace = figure('Name','jspace');
set(fig_jspace,'color','w');
ax_j = axes(fig_jspace,'View',[0 90]);
axis equal
hold on
ax_j.XLim = [-pi pi];
ax_j.YLim = [-pi pi];
ax_j.ZLim = [-0.1 0.1];
n_grid = 50;
x_span = linspace(-pi, pi, n_grid);
y_span = linspace(-pi, pi, n_grid);
[X_mg,Y_mg] = meshgrid(x_span, y_span);
q=[X_mg(:) Y_mg(:)];
tic
[dst, rep] = getClosestDistanceVec(q, y_pos, link_sym, r, d, alpha, base, n_pts);
Z_mg = reshape(dst,size(X_mg));
[~, ctr_j] = contourf(ax_j, X_mg,Y_mg,Z_mg,100,'LineStyle','none');
toc

%%

function [dist, rep] = getClosestDistance(j_state, y_pos, link_sym, r,d,alpha,base, n_pts)
    tic
    link_num = numeric_fk_model(j_state,r,d,alpha,base, y_pos, n_pts);
    ii = link_num{1}.minidx(1);
    jj = link_num{1}.minidx(2);
    pt_closest = link_num{ii}.pts(jj,:);
    dist = link_sym{ii}.dst_fcn(j_state', y_pos, pt_closest');
    rep  = link_sym{ii}.rep_fcn(j_state', y_pos, pt_closest');
    toc
end

function [dist, rep] = getClosestDistanceVec(j_state, y_pos, link_sym, r,d,alpha,base, n_pts)
    k = 1;
    for i = 1:1:size(y_pos,2) 
        for j = 1:1:size(j_state,1)
            [dist(k), rep(k,:)] = getClosestDistance(j_state(j,:), y_pos(:,i), link_sym, r, d, alpha, base, n_pts);
            k=k+1;
        end
    end
end

function link = numeric_fk_model(j_state,r,d,alpha,base,y, N_pts)
    DOF = length(r)-1;
    vecspace = @(v1,v2,k) v1+linspace(0,1,k)'.*(v2-v1);
    P = dh_fk(j_state,r,d,alpha,base);
    dist = @(x,y) sqrt((x-y)'*(x-y));
    ddist = @(x,y) 1/sqrt((x-y)'*(x-y))*[x(1)-y(1);x(2)-y(2); x(3)-y(3)];
    link{1}.minidx = [0 0];
    link{1}.mindist = 1000;
    for i = 1:1:DOF
        link{i}.pts = vecspace([0 0 0], [r(i) 0 0], N_pts);
        link{i}.R = P{i+1}(1:3,1:3);
        link{i}.T = P{i+1}(1:3,4);
        for j = 1:1:size(link{i}.pts, 1)
            v = link{i}.pts(j,:);
            R = link{i}.R;
            T = link{i}.T;
            pos = v*R'+T';
            link{i}.pos(j,:) = pos;
            link{i}.dist(j) = dist(pos', y);
            if link{i}.dist(j)<=link{1}.mindist
                link{1}.mindist = link{i}.dist(j);
                link{1}.minidx = [i j];
            end
        end
    end
end


function link = symbolic_fk_model(j_state_sym,r,d,alpha,base,y_sym, p_sym)
    DOF = length(r)-1;
    P = dh_fk(j_state_sym,r,d,alpha,base);
    dist = @(x,y) sqrt((x-y)'*(x-y));
    ddist = @(x,y) 1/sqrt((x-y)'*(x-y))*[x(1)-y(1);x(2)-y(2); x(3)-y(3)];
    for i = 1:1:DOF
        link{i}.R = P{i+1}(1:3,1:3);
        link{i}.T = P{i+1}(1:3,4);
        R = link{i}.R;
        T = link{i}.T;
        pos = p_sym'*R'+T';
        link{i}.pos = pos;
        link{i}.dist = dist(pos', y_sym);
        J_TMP = jacobian(pos,j_state_sym);
        J_TMP2 = sym(zeros(size(y_sym,1),DOF));
        J_TMP2(1:size(J_TMP,1),1:1:size(J_TMP,2)) = J_TMP;
        link{i}.rep = ddist(pos',y_sym)' * J_TMP2;
        % get quick handlers
        link{i}.dst_fcn = matlabFunction(link{i}.dist,'Vars',{j_state_sym,y_sym,p_sym});
        link{i}.rep_fcn = matlabFunction(link{i}.rep,'Vars',{j_state_sym,y_sym,p_sym});
    end
end


function handle = create_r(ax_h, link_num)
    pts = link_num{1}.pos(1, :);
    for i = 1:1:length(link_num)
        pts = [pts; link_num{i}.pos(end, :)];
    end
    handle = plot(ax_h,pts(:,1),pts(:,2),'LineWidth',2,...
        'Marker','o','MarkerFaceColor','k','MarkerSize',4);
end

function update_r(r_handle,link_num)
    pts = link_num{1}.pos(1, :);
    for i = 1:1:length(link_num)
        pts = [pts; link_num{i}.pos(end, :)];
    end
    r_handle.XData = pts(:,1);
    r_handle.YData = pts(:,2);
end


function pts = calc_fk(j_state,r,d,alpha,base)
    P = dh_fk(j_state,r,d,alpha,base);
    pts = zeros(3,3);
    for i = 1:1:length(j_state)+1
        v = [0,0,0];
        R = P{i}(1:3,1:3);
        T = P{i}(1:3,4);
        p = v*R'+T';
        pts(i,:) = p;
    end
end
