%% s = logP = MCMC_logP(residual_sos, noise, Nm)
%
% Input
% --------------
%
% Output
% --------------
%
% Description: Support function for gpuAxCaliberSMTmcmc to be able to use arrayfunc
%              Compute the log probability
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 22 March 2024
% Date modified: 14 Nov 2024
%
%
function lp = logP_Gaussian(residual_sos, noise, Nm)

lp = -residual_sos./(2*noise.^2) - Nm/2*log((2*pi*noise.^2));

end