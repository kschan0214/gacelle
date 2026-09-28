function xt = thin_by_ess(x)
% THIN_BY_ESS Thin one chain so that retained samples are roughly independent.
%
%   xt = thin_by_ess(x)
%
% x  : samples of one voxel/parameter, vector (all chains concatenated is
%      NOT allowed; pass one chain)
% xt : x(1:stride:end), stride = max(1, ceil(2*tau)), tau = N/ESS, with
%      ESS from mcmc_bayes.ess (single chain, split in two)
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%
x      = x(:).';
tau    = numel(x) / mcmc_bayes.ess(x);
stride = max(1, ceil(2*tau));
xt     = x(1:stride:end);
end
