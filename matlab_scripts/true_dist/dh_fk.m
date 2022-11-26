function P = dh_fk(j_state,a,d,alpha,base)
    A = @(a,d,alpha,q) [cos(q) -sin(q)*cos(alpha)   sin(q)*sin(alpha)	a*cos(q);
                        sin(q)	cos(q)*cos(alpha)	-cos(q)*sin(alpha)  a*sin(q);
                        0       sin(alpha)          cos(alpha)          d;
                        0       0                   0                   1];
    Am =@(a,d,alpha,q) [cos(q) -sin(q) 0 a;
                       sin(q)*cos(alpha) cos(q)*cos(alpha) -sin(alpha) -d*sin(alpha);
                       sin(q)*sin(alpha) cos(q)*sin(alpha) cos(alpha) d*cos(alpha);
                       0    0   0   1];
    %base matrix holds index 1 instead of 0 (because matlab)
    P{1} = base;
    %kinematic chain
    for i = 2:1:length(j_state)+1
        P{i} = P{i-1}*Am(a(i-1),d(i-1),alpha(i-1),j_state(i-1));
    end
end
