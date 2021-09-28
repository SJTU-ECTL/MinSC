function y = Bern(x, k, n)
y = nchoosek(n, k) * x.^k .* (1-x).^(n-k);
% end of function Bern
