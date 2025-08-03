% Generate RPCA problem D=L+S; size = n, rank(L) = r, density(S) = alpha
function [L, D] = rpca_tensor(n,r,alpha)
if length(n) ~= length(r)
    disp('Mode of tensor and Tucker rank must agree')
end
L = randn(r);
L = tensor(L);
ndim = length(n);
for i = 1:ndim
    L = ttm(L,randn(n(i),r(i)),i);
end
% generate sparse tensor S
S_supp_idx = randsample(prod(n), round(alpha*prod(n)), false);
S_range = mean(abs(L(:)));
S_temp = 2*S_range*rand(n)-S_range; 
S = zeros(n);
S(S_supp_idx) = S_temp(S_supp_idx);
S = tensor(S);    
D = L + S;
end