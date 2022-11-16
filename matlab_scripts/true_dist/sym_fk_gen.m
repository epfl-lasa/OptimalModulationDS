
%%
r = [3 3 0];
d = [0 0 0];
alpha = [0 0 0];
base = eye(4);
syms j_state_sym [length(r)-1 1]
syms y_sym [3, 1] %planar => z=0
syms p_sym [3, 1]
%%

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


