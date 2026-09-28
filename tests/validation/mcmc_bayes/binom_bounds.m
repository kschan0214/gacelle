function [lo, hi] = binom_bounds(n, p, alpha)
% BINOM_BOUNDS Exact two-sided (1-alpha) central interval of Binomial(n,p).
%
%   [lo, hi] = binom_bounds(n, p, alpha)
%
% lo : largest k with P(X < k) <= alpha/2 (i.e. P(X <= lo-1) <= alpha/2)
% hi : smallest k with P(X > k) <= alpha/2
% Computed with gammaln, no Statistics toolbox.
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%
k    = (0:n)';
logp = gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1) + k*log(p) + (n-k)*log1p(-p);
pmf  = exp(logp);
cdf  = cumsum(pmf);
lo   = find(cdf > alpha/2, 1) - 1;          % P(X <= lo-1) <= alpha/2
hi   = find(1 - cdf <= alpha/2, 1) - 1;     % P(X > hi) <= alpha/2
end
