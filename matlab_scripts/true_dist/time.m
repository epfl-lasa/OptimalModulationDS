clc
clear all
close all force
vecspace = @(v1,v2,k) v1+linspace(0,1,k)'.*(v2-v1);
global all_state
%rng(1)
%%
dh_r = [0 3 3 ];
d = dh_r*0;
alpha = dh_r*0;
base = eye(4);
DOFs = length(dh_r)-1;
syms j_state_sym [DOFs 1]
syms y_sym [3, 1] %planar => z=0
syms p_sym [3, 1]
p_s = zeros(DOFs,1);
p_s(1) = pi/2;
p_f = zeros(DOFs,1);
p_f(1) = -pi/2;
q_min = -pi*ones(1, DOFs);
q_max = pi*ones(1, DOFs);
y_min = [-10, -10];
y_max = [10, 10];
state_min = [q_min, y_min, 0.01];
state_max = [q_max, y_max, 10];
%% numeric & symbolic models
n_pts = 20;
link_sym = symbolic_fk_model(j_state_sym,dh_r,d,alpha,base, y_sym, p_sym);
j_state = p_s;
y_pos = [5; 0; 0];
y_r = 1;
link_num = numeric_fk_model(j_state,dh_r,d,alpha,base, y_pos, n_pts);
all_state = [j_state; y_pos(1:2); y_r];
tic;
for i = 1:1:1000
    link_num = numeric_fk_model(j_state,dh_r,0*dh_r,0*dh_r,eye(4), y_pos, 10);
    dst_coll = link_num{1}.mindist - y_r;
    %[dst, n_v] = getClosestDistance(j_state, y_pos, link_sym, dh_r,0*dh_r,0*dh_r,eye(4), 10);
end
toc
%% functions 
function [traj, dst_arr, ker_val] = propagate_mod(pol, j_state, q_f, dt, N, dh_r, y_pos, y_r, link_sym, q_min, q_max)
    rbf = @(x, x0, s)exp(-norm(x-x0)^2/(2*s^2));
    rotm = @(ang)[cos(ang) sin(ang);
             -sin(ang) cos(ang)];
    A = -1*eye(length(j_state));
    q_cur = j_state;
    traj = zeros(length(j_state), N);
    n_ker = length(pol.sigma);
    ker_val = zeros(n_ker, N);
    dst_arr  = 100+zeros(1,N);
    for i = 1:1:N
%         if i == 26 %debug stopping
%             i
%         end
        %nominal motion
        q_dot = A*(q_cur-q_f);
        %current distance
        [dst, n_v] = getClosestDistance(q_cur', y_pos, link_sym, dh_r,0*dh_r,0*dh_r,eye(4), 10);
        dst = dst - y_r - 0.;
        dst_arr(i) = dst;
        % modulation
        %tau_v = [n_v(2), -n_v(1)];
        B = eye(length(n_v));
        B(:,1) = n_v';
        E = gs_m(B);
        g = max(1e-8, dst+1);
        l_n = max(0, 1 - 1/g);
        l_tau = max(1, 1 + 1/g);
        %E = [n_v' tau_v'];
        D = l_tau*eye(length(n_v));
        D(1,1) = l_n;
        %D = eye(2);
        u_cur = 0;
        for j = 1:1:n_ker
            ker_val(j, i) = rbf(q_cur,pol.x0(:,j),pol.sigma(j));
            u_cur = u_cur + pol.alpha(:,j)/norm(pol.alpha(:,j))*ker_val(j, i);
        end
        q_dot = E*D*E' * (q_dot + sum(u_cur'.*E(:,2:end),2) + (1-l_n)*E(:,1));

        %q_dot = E*D*E' * alpha* q_dot +E*D*E' * 1-alpha v(i) E*D*E' * 1-alpha v(i);
        %slow for collision
        if dst < 0
            q_dot = q_dot*0.1;
        end
        if norm(q_dot)>1e-1
            q_dot = q_dot/norm(q_dot);
        end
        %slow for joint_limits
        if any(q_cur<=q_min') || any(q_cur>=q_max')
            q_dot = q_dot*0.1;
        end
        q_cur = q_cur + q_dot * dt;
        %q_cur = min(max(q_cur, q_min'), q_max');

        traj(:, i) = q_cur;
    end    
end


function slider_change(~, event, i, dh_r, link_sym, ...
                    h_rob, ctr_r, ctr_j, ctr_j_boundary, ...
                    h_circ, h_j_dot, state_min, state_max)
    global all_state
    %apply change
    all_state(i) = event.Value;

    j_state = all_state(1:2);
    y_pos = [all_state(3:4); 0];
    y_r = all_state(end);
    link_num = numeric_fk_model(j_state,dh_r,0*dh_r,0*dh_r,eye(4), y_pos, 10);
    

    %moving the robot
    update_r(h_rob, link_num)
    
    %moving the circle
    [xc, yc] = circle_pts(y_pos(1), y_pos(2), y_r);
    h_circ.XData = xc;
    h_circ.YData = yc;
    
    %moving jspace point
    h_j_dot.XData = j_state(1);
    h_j_dot.YData = j_state(2);

    %distfields calc
    if strcmp(event.EventName, 'ValueChanged')
        %robot distfield
%         x_span = linspace(state_min(3), state_max(3), size(ctr_r.XData,1));
%         y_span = linspace(state_min(4), state_max(4), size(ctr_r.YData,1));
%         [X_mg,Y_mg] = meshgrid(x_span, y_span);
%         y=[X_mg(:) Y_mg(:) 0*Y_mg(:)]';
%         [dst, ~] = getClosestDistanceVec(j_state', y, link_sym, ...
%                                 dh_r,0*dh_r,0*dh_r,eye(4), 10);
%         Z_mg = reshape(dst,size(X_mg));
%         ctr_r.ZData = Z_mg;
%         ctr_r.LevelList=linspace(min(dst),max(dst));

        %jointspace distfield
        x_span = linspace(state_min(1), state_max(1), size(ctr_j.XData,1));
        y_span = linspace(state_min(2), state_max(2), size(ctr_j.YData,1));
        [X_mg,Y_mg] = meshgrid(x_span, y_span);
        q=[X_mg(:) Y_mg(:)];
        [dst, ~] = getClosestDistanceVec(q, y_pos, link_sym, ...
                                dh_r,0*dh_r,0*dh_r,eye(4),10);
        dst = dst - y_r;
        Z_mg = reshape(dst,size(X_mg));
        ctr_j.ZData = Z_mg;
        ctr_j_boundary.ZData = Z_mg;
        ctr_j.LevelList=linspace(min(dst),max(dst));
    end
    drawnow
end

function [dist, rep] = getClosestDistance(j_state, y_pos, link_sym, r,d,alpha,base, n_pts)
    link_num = numeric_fk_model(j_state,r,d,alpha,base, y_pos, n_pts);
    ii = link_num{1}.minidx(1);
    jj = link_num{1}.minidx(2);
    if ii==1 && jj == 1
        jj = 2;
    end
    pt_closest = link_num{ii}.pts(jj,:);
    dist = link_sym{ii}.dst_fcn(j_state', y_pos, pt_closest');
    rep  = link_sym{ii}.rep_fcn(j_state', y_pos, pt_closest');
    if norm(rep)>0
        rep = rep/norm(rep);
    end
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
    vecspace = @(v1,v2,k) v1+linspace(0,1,max(2,k))'.*(v2-v1);
    P = dh_fk(j_state,r,d,alpha,base);
    dist = @(x,y) sqrt((x-y)'*(x-y));
    ddist = @(x,y) 1/sqrt((x-y)'*(x-y))*[x(1)-y(1);x(2)-y(2); x(3)-y(3)];
    link{1}.minidx = [0 0];
    link{1}.mindist = 1000;
    for i = 1:1:DOF
        link{i}.pts = vecspace([0 0 0], [r(i+1) 0 0], N_pts);
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

function [xc,yc] = circle_pts(x,y,r)
    hold on
    th = linspace(0,2*pi,50);
    xc = r * cos(th) + x;
    yc = r * sin(th) + y;
end
