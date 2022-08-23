syms n1 n2 t1 t2 l1 l2 p1 p2 x1 x2
M = [n1 n2; t1 t2]' * [l1 0 ; 0 l2] * [n1 n2; t1 t2];
f = [-p1*x1; -p2*x2];
M*f
