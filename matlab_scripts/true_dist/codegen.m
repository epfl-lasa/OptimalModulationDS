clc
clear all
close all force
vecspace = @(v1,v2,k) v1+linspace(0,1,k)'.*(v2-v1);
global all_state
%rng(1)
%%
DOFs = 7;
syms j_state_sym [DOFs 1]
syms y_sym [3, 1] %planar => z=0
syms p_sym [3, 1]
syms dh_r [DOFs 1]
d = dh_r*0;
alpha = dh_r*0;
base = eye(4);

%% numeric & symbolic models
n_pts = 20;
link_sym = symbolic_fk_model(j_state_sym,dh_r,d,alpha,base, y_sym, p_sym);
%% functions 


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
        pos = simplify(p_sym'*R'+T');
        link{i}.pos = pos;
        link{i}.dist = simplify(dist(pos', y_sym));
        J_TMP = jacobian(pos,j_state_sym);
        J_TMP2 = sym(zeros(size(y_sym,1),DOF));
        J_TMP2(1:size(J_TMP,1),1:1:size(J_TMP,2)) = J_TMP;
        link{i}.rep = simplify(ddist(pos',y_sym)' * J_TMP2);
    end
end
