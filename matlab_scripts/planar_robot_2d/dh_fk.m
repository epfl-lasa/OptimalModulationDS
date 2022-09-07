function P = dh_fk(j_state,r,d,alpha,base)
    Am = @(r,d,alpha,q) [cos(q) -sin(q)*cos(alpha)   sin(q)*sin(alpha)	r*cos(q);
                        sin(q)	cos(q)*cos(alpha)	-cos(q)*sin(alpha)  r*sin(q);
                        0       sin(alpha)          cos(alpha)          d;
                        0       0                   0                   1];
    A =@(r,d,alpha,q) [cos(q) -sin(q) 0 r;
                       sin(q)*cos(alpha) cos(q)*cos(alpha) -sin(alpha) -d*sin(alpha);
                       sin(q)*sin(alpha) cos(q)*sin(alpha) cos(alpha) d*cos(alpha);
                       0    0   0   1];
    %base matrix holds index 1 instead of 0 (because matlab)
    P{1} = base;
    %kinematic chain for 3 joints
    for i = 2:1:3
        P{i} = P{i-1}*Am(r(i-1),d(i-1),alpha(i-1),j_state(i-1));
    end
end
