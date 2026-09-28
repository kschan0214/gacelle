function [z, qS, se, essQ] = quantile_z(xs, qRef, fRef, eRef, qs)
% QUANTILE_Z MC z-scores of sampler quantiles against reference quantiles.
%
%   [z, qS, se, essQ] = quantile_z(xs, qRef, fRef, eRef, qs)
%
% xs    : samples of one voxel/parameter, [1, Ns, Nchain]
% qRef  : reference quantiles (1 x Nq)
% fRef  : reference density at qRef
% eRef  : reference error (grid discretisation or reference MC), 1 x Nq
% qs    : probabilities
% z     : (qS - qRef) ./ sqrt(se.^2 + eRef.^2)
% qS    : sampler quantiles of the pooled chains (order statistic ceil(q*N))
% se    : MC standard error of qS: sqrt(q(1-q)/ESS_q) / fRef, with ESS_q the
%         multi-chain ESS (mcmc_bayes.ess) of the indicator 1{x <= qRef}
%         (the quantile SE of Vehtari et al. 2021, density from the reference)
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%
xs   = double(xs);
xv   = sort(xs(:));
N    = numel(xv);
qS   = zeros(size(qs)); se = qS; essQ = qS;
for k = 1:numel(qs)
    qS(k)   = xv(max(1, ceil(qs(k)*N)));
    I       = double(xs <= qRef(k));
    e       = mcmc_bayes.ess(I);
    if ~(e > 0) || ~isfinite(e); e = mcmc_bayes.ess(xs); end   % indicator constant in all chains
    essQ(k) = min(e, N);
    se(k)   = sqrt(qs(k)*(1-qs(k))/essQ(k)) / fRef(k);
end
z = (qS - qRef) ./ sqrt(se.^2 + eRef.^2);
end
