clc
clear all
close all force
vecspace = @(v1,v2,k) v1+linspace(0,1,k)'.*(v2-v1);
global all_state
%rng(1)
%%
dh_r = [0 1 1 1 1 1];
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
%% task space plot
fig_robot = figure('Name','Two-dimensional robot','Position',[100 100 450 353]);
set(fig_robot,'color','w');
ax_r = axes(fig_robot,'View',[0 90]);
axis equal
hold on
ax_r.XLim = [y_min(1) y_max(1)];
ax_r.YLim = [y_min(2) y_max(2)];
ax_r.ZLim = [-0.1 0.1];
xlabel('x')
ylabel('y')

%plot(ax_h, link_num{1}.pos(:,1), link_num{1}.pos(:,2), 'r*')
ctr_r = 0;
h_rob = create_r(ax_r, link_num);

link_num = numeric_fk_model(j_state,dh_r,d,alpha,base, y_pos, n_pts);

update_r(h_rob, link_num)

[xc, yc] = circle_pts(y_pos(1), y_pos(2), y_r);
h_kcirc = plot(ax_r, xc, yc, 'r-','LineWidth',1.5);

%% joint space plot
fig_jspace = figure('Name','jspace','Position',[500 100 450 353]);
set(fig_jspace,'color','w');
ax_j = axes(fig_jspace,'View',[0 90]);
axis equal
hold on
ax_j.XLim = [-pi pi]*1.1;
ax_j.YLim = [-pi pi]*1.1;
ax_j.ZLim = [-0.1 0.1];
n_grid = 30;
x_span = linspace(-pi, pi, n_grid);
y_span = linspace(-pi, pi, n_grid);
[X_mg,Y_mg] = meshgrid(x_span, y_span);
q = zeros(length(X_mg(:)), DOFs);
q(:,1)=X_mg(:);
q(:,2)=Y_mg(:);
xlabel('q1')
ylabel('q2')
%tic
[dst, rep] = getClosestDistanceVec(q, y_pos, link_sym, dh_r, d, alpha, base, n_pts);
dst = dst - y_r;
Z_mg = reshape(dst,size(X_mg));
[~, ctr_j] = contourf(ax_j, X_mg,Y_mg,Z_mg,100,'LineStyle','none');
[~, ctr_j_bndr] = contour(ax_j, X_mg,Y_mg,Z_mg,[0, 0+1e-3],'LineWidth',2,'Color','k');
%toc

%% motion
plot(ax_j, p_f(1), p_f(2), 'r*')
N_STEPS = 50;
dt = 0.2;
N_TRAJ = 10;

%kernel sampling parameters
N_KER = 0;
N_KER_MAX = 50; 

MU_C = zeros(DOFs, N_KER);
S_NOMINAL = 0.03 * max(q_max-q_min); 
MU_S = S_NOMINAL*ones(1, N_KER);
MU_A = zeros(DOFs-1,N_KER); %alphas for tangential space
SIGMA_C  = 0.0;
SIGMA_S = 0.0;
SIGMA_A = 0.3;

%plotting handlers
h_traj = cell(1, N_TRAJ);
h_ker = cell(1, N_KER_MAX);
h_kcirc = cell(1, N_KER_MAX);
for i = 1:1:N_TRAJ
    h_traj{i} = plot(ax_j, 100, 100, 'b-');
end

[xc, yc] = circle_pts(100, 100, 1);
for i = 1:1:N_KER_MAX
    h_ker{i} = plot(ax_j, 100, 100, 'c.','MarkerSize',2);
    h_kcirc{i} = plot(ax_j, xc, yc, 'c-','LineWidth',1.5);
end
h_j_pos = plot(ax_j, j_state(1), j_state(2), 'g*');

%main loop
MAIN_ITER = 1;
ker_added = 0;
while norm(j_state-p_f)>1e-1
    tic
    link_num = numeric_fk_model(j_state,dh_r,0*dh_r,0*dh_r,eye(4), y_pos, 10);
    dst_coll = link_num{1}.mindist - y_r;
    dst_mu = 100;
    for i = 1:1:N_KER
        dst_mu = min(dst_mu, norm(j_state-MU_C(:,i)));
    end
    msg = sprintf('Iter: %d, Collision dist: %4.2f, Kernel dist: %4.2f', ...
        MAIN_ITER, dst_coll, dst_mu);
    disp(msg)
    %if we add kernel based on previous rollouts
    if ker_added
        [~, closest_ker_idx] = min(vecnorm(MU_C - new_ker));
        N_KER = N_KER + 1;
        MU_C = [MU_C new_ker];
        MU_S = [MU_S S_NOMINAL];
        MU_A = [MU_A zeros(DOFs-1,1)];
        if N_KER>1
            MU_A(end) = MU_A(closest_ker_idx);
        end
    end
    %sample policies
    centers = normrnd(repmat(MU_C,[1,1,N_TRAJ]),SIGMA_C); %centers of rbfs
    sigmas = normrnd(repmat(MU_S,[N_TRAJ,1]),SIGMA_S); %width of rbfs
    alphas = normrnd(repmat(MU_A,[1,1,N_TRAJ]),SIGMA_A); %amplitude of rbfs

    %add seed directions
    if N_KER>0
        tmpl = eye(DOFs-1);
        for i = 1:1:DOFs-1
            %first positive tangent in the beginning
            alphas(:, :, i)     = alphas(:, :, 1)*0+tmpl(i,:)';
            %then negative tangent in the end
            alphas(:, :, end-i+1) = alphas(:, :, 1)*0-tmpl(i,:)';
        end
    end
    
    for i = 1:1:N_TRAJ
        pol{i}.x0 = squeeze(centers(:,:,i));
        pol{i}.sigma = sigmas(i,:);
        pol{i}.alpha = squeeze(alphas(:,:,i));
    end
    %integration
    %tic
    %ker_use = zeros(N_TRAJ, N_KER);
    for i = 1:1:N_TRAJ
        [traj{i}, dist{i}, ker_val{i}] = propagate_mod(pol{i},j_state, p_f, dt, N_STEPS, dh_r, y_pos, y_r, link_sym, q_min, q_max);
        traj{i} = [j_state traj{i}];
    end
    %toc
    
    %cost calculation
    cost = zeros(1, N_TRAJ);
    for i = 1:1:N_TRAJ
        %1) reaching goal cost
        goal_cost_tmp = norm(traj{i}(:,end)-p_f); 
        %2) collision cost
        coll_cost_tmp = sum(dist{i}<0)*100;
        %3) joint limits cost
        j_lim_cost = sum(any(traj{i}<q_min'))*100+sum(any(traj{i}>q_max'))*100;
        %4) trajectory length cost (not to stay or oscillate)
        stay_cost = 100*goal_cost_tmp * 1/norm(traj{i}(:,1)-traj{i}(:,end));
        cost(i) = goal_cost_tmp+coll_cost_tmp+j_lim_cost+stay_cost;
    end
    
    %add new kernels
    ker_added = 0;
    %find all points close-to-collision and far from existing kernels
    potential_ker = zeros(0,2);
    ker_use = zeros(N_TRAJ, N_KER);
    for i = 1:1:N_TRAJ
        ker_use(i,:) = sum(ker_val{i},2)';
        %indices of trajectory points close-to-collision
        idx_close = (dist{i}>0) & (dist{i}<0.5);
        %indices of trajectory points away from existing kernels
        idx_no_ker = zeros(size(idx_close));
%         idx_no_ker(all(abs(ker_val{i})<0.5,1)) = 1;
        idx_no_ker(sum(abs(ker_val{i}),1)<0.3) = 1;

        %potential new kernel locations
        potential_ker = [potential_ker traj{i}(:, idx_close & idx_no_ker)];
%         if sum(idx_close & idx_no_ker)>0
%             i
%         end
    end
    potential_ker = unique(potential_ker','rows')';
    if size(potential_ker,2)>0
        idx=randperm(size(potential_ker,2),1);
        new_ker=potential_ker(:,idx);
        ker_added = 1;
        MAIN_ITER 
    end
    %plotting
    for i = 1:1:N_TRAJ
        h_traj{i}.XData =  traj{i}(1,:);
        h_traj{i}.YData =  traj{i}(2,:);
    end
    
    [mval, midx] = min(cost);
    if var(cost)>0.1
        for i = 1:1:N_KER
            h_ker{i}.XData = pol{midx}.x0(1, i);
            h_ker{i}.YData = pol{midx}.x0(2, i);
            [xc, yc] = circle_pts(pol{midx}.x0(1, i), pol{midx}.x0(2, i), 3*pol{midx}.sigma(i));
            h_kcirc{i}.XData = xc;
            h_kcirc{i}.YData = yc;
            if pol{midx}.alpha(i)>0
                h_kcirc{i}.Color = [1 1 0];
            else
                h_kcirc{i}.Color = [0 1 1];
            end
        end
    end
    %moving the robot
    x_dot = diff(traj{midx}(:,1:2),1,2);
    if norm(x_dot)>1e-8
        x_dot = x_dot/norm(x_dot);
    end
    j_state = j_state + 0.1*x_dot;
    h_j_pos.XData = j_state(1);
    h_j_pos.YData = j_state(2);

    link_num = numeric_fk_model(j_state,dh_r,0*dh_r,0*dh_r,eye(4), y_pos, 5);
    update_r(h_rob, link_num)
    drawnow

    %UPDATE POLICY
    beta = mean(cost)/50;
    w = exp(-1/beta * cost);
    w = w/sum(w);

    k_act = sum(ker_use);
    w_act = exp(1/mean(k_act) * k_act);
    %UPD_RATE = w_act/sum(w_act);
    UPD_RATE = 0.6;
    UPD_RATE = 0.6*ones(1,N_KER);
    UPD_RATE(k_act<0.1) = 0;
    w_e(1,1,:) = w;
    MU_C = (1-UPD_RATE).*MU_C + UPD_RATE.*sum(w_e .* centers,3);
    MU_A = (1-UPD_RATE).*MU_A + UPD_RATE.*sum(w_e .* alphas, 3);
    MU_S = (1-UPD_RATE).*MU_S + UPD_RATE.*sum(w'.*sigmas);
    MAIN_ITER = MAIN_ITER+1;
    toc
    1/toc
end

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
        q_dot_nom = A*(q_cur-q_f);
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
        q_dot = E*D*E' * (q_dot_nom + sum(u_cur'.*E(:,2:end),2) + (1-l_n)*E(:,1));
        %q_dot = E*D*E' * q_dot_nom;
        %q_dot = E*D*E' * alpha* q_dot +E*D*E' * 1-alpha v(i) E*D*E' * 1-alpha v(i);
        if norm(q_dot)>1e-1
            q_dot = q_dot/norm(q_dot);
        end
        %slow for collision
        if dst < 0
            q_dot = q_dot*0.1;
        end
        %slow for joint_limits
        if any(q_cur<=q_min') || any(q_cur>=q_max')
            q_dot = q_dot*0.1;
        end
        q_cur = q_cur + q_dot * dt;
        %q_cur = min(max(q_cur, q_min'), q_max');

        traj(:, i) = q_cur;
        %q_cur
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
%     if ii==1 && jj == 1
%         jj = 2;
%     end
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
    vecspace = @(v1,v2,k) v1+linspace(0.01,1,max(2,k))'.*(v2-v1);
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
