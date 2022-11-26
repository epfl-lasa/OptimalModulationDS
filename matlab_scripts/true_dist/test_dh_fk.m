dh_r = [0 3 3];
d = dh_r*0;
alpha = dh_r*0;
base = eye(4);

q = [0 0];

P = dh_fk(q, dh_r, d, alpha, base);
for i = 1:1:length(P)
    P{i}
end