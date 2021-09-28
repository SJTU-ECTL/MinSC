% This function finds the closet Bernstein polynomial approximation of degree n
% to the given function funchandle.
%
% It optimizes the objective function obj = coefs'*H*coefs + 2*b'*coefs + c
% subject to 0 <= coefs <=1, where coefs is the vector of Bernstein
% coefficients to be solved.
% Matrix H and vector b correspond to the matrix H and vector c defined in the
% Section 2.2 of the paper "An Architecture for Fault-Tolerant Computation with
% Stochastic Logic". Constant c is the integral of func_to_appr*func_to_appr
% over the given range [int_l, int_u].
%
% Usage: [coefs, obj, H, b, c] = Bern_appr(funchandle, n)
% funchandle is provided in func_to_appr.m and n is the degree of the
% Bernstein polynomial for approximation.
% Example: [coefs, obj] = Bern_appr(@func_to_appr, 6)
%
function [coefs, obj, H, b, c] = Bern_appr(funchandle, n)
tol = 1.e-12;
int_l = 0;
int_u = 1;

% set up the matrix H
for i = 1:(n+1)
	for j = i:(n+1)
		fhandle = @(x)(Bern(x, i-1, n).*Bern(x, j-1, n));
		H(i,j) = quad(fhandle, int_l, int_u, tol);
		if i ~= j
			H(j,i) = H(i,j);
		end
	end
end

% set up the vector b
for i= 1:(n+1)
	fhandle=@(x)(Bern(x, i-1, n).*funchandle(x));
	b(i,1) = -quad(fhandle, int_l, int_u, tol);
end

% set up the constant c
fhandle = @(x)(funchandle(x).*funchandle(x));
c = quad(fhandle, int_l, int_u, tol);

options = optimset('LargeScale','off');

lb = zeros(n+1,1);
ub = ones(n+1,1);
x0 = [];
[coefs, val, exitflags] = quadprog(H,b,[],[],[],[],lb,ub,x0,options);
obj = coefs' * H * coefs + 2 * b' * coefs + c;

