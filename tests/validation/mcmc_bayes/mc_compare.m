function [zMean, zSD] = mc_compare(xA, xB)
% MC_COMPARE Monte Carlo z-scores of posterior mean and SD between two runs.
%
%   [zMean, zSD] = mc_compare(xA, xB)
%
% xA, xB : samples, [Nv, Ns, Nchains] (native space)
% zMean  : (mA - mB) / sqrt(seA^2 + seB^2),  se = sd / sqrt(ESS(x))
% zSD    : (sA - sB) / sqrt(seA^2 + seB^2),  se(sd) = sqrt(var((x-m)^2)/ESS((x-m)^2)) / (2 sd)
% ESS is mcmc_bayes.ess (split chains, Geyer initial monotone sequence).
% Under equal posteriors, both are approximately N(0,1) per voxel.
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%
[mA, sA, seMA, seSA] = moments_se(xA);
[mB, sB, seMB, seSB] = moments_se(xB);
zMean = (mA - mB) ./ sqrt(seMA.^2 + seMB.^2);
zSD   = (sA - sB) ./ sqrt(seSA.^2 + seSB.^2);
end

function [m, s, seM, seS] = moments_se(x)
x    = double(x);
Nv   = size(x,1);
xf   = reshape(x, Nv, []);
m    = mean(xf, 2);
s    = std(xf, 0, 2);
seM  = s ./ sqrt(mcmc_bayes.ess(x));
d2   = (x - m).^2;
seV  = sqrt(var(reshape(d2, Nv, []), 0, 2) ./ mcmc_bayes.ess(d2));
seS  = seV ./ (2*s);
end
