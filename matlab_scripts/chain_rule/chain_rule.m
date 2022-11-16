clc
clear all

r = [3 3 0];
d = [0 0 0];
alpha = [0 0 0];
base = eye(4);
syms j_state [2 1]
syms x [2,1] 
syms y [2,1]

Am = @(r,d,alpha,q) [cos(q) -sin(q)*cos(alpha)   sin(q)*sin(alpha)	r*cos(q);
                    sin(q)	cos(q)*cos(alpha)	-cos(q)*sin(alpha)  r*sin(q);
                    0       sin(alpha)          cos(alpha)          d;
                    0       0                   0                   1];
%base matrix holds index 1 instead of 0 (because matlab)
P{1} = base;
%kinematic chain for 3 joints
for i = 2:1:3
    P{i} = P{i-1}*Am(r(i-1),d(i-1),alpha(i-1),j_state(i-1));
end
R_ee = P{3}(1:3,1:3);
T_ee = P{3}(1:3,4);
tmp_pos_ee = [0 0 0]*R_ee'+T_ee';
pos_ee = tmp_pos_ee(1:2)';
j_ee = jacobian(pos_ee);
%subs(pos_ee, j_state, [0;0])
%subs(j, j_state, [0;0])
dpos = @(x,y) 1/sqrt((x-y)'*(x-y))*[x(1)-y(1);x(2)-y(2)];
n_sym = (dpos(pos_ee,y)' * j_ee)';
j_state_val = [0 0]';
y_pos_val = [6 2]';
ee_pos = double(subs(pos_ee,j_state,j_state_val));
n = double(subs(n_sym,[j_state, y],[j_state_val, y_pos_val]));

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

function handle = create_r(ax_h,j_state,r,d,alpha,base)
    pts = calc_fk(j_state,r,d,alpha,base);
    handle = plot(ax_h,pts(:,1),pts(:,2),'LineWidth',2,...
        'Marker','o','MarkerFaceColor','k','MarkerSize',4);
end

function move_r(r_handle,j_state,r,d,alpha,base)
    pts = calc_fk(j_state,r,d,alpha,base);
    r_handle.XData = pts(:,1);
    r_handle.YData = pts(:,2);
end

