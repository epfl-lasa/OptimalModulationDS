clc
clear all

A = sym('A',[2,2]);

B = sym('B',[2,2]);
B(1,2) = 0;
B(2,1) = 0;
