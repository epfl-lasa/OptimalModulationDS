import sympy as sp
n = 2
A = sp.MatrixSymbol('A', n, n)
B = sp.MatrixSymbol('B', n, n)
C = sp.MatrixSymbol('C', n, n)
D = sp.MatrixSymbol('D', n, n)

M = sp.Matrix([[A, B], [C, D]])
for i in range(5):
    print('Power %d' % i)
    print(M**i)
