function y = Bern1(x, k, n)
y = x.^k .* (1-x).^(n-k);
% end of function Bern