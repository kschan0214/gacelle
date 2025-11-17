%% s = diffusion_relaxation_spherical_mean_combine(f, D, R2a, R2e, b,te)
%
% Input
% --------------
%
% Output
% --------------
%
% Description: Support function for gpuAxCaliberSMTmcmc to be able to use arrayfunc
%              Compute the combined dMRI signal ffrom all compartments
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 12 November 2025
% Date modified: 
%
%
function s = diffusion_relaxation_spherical_mean_combine(f, D, R2a, R2e, b,te)

% avoid division by zeros
b = max(b,1e-10);

% 1st order tortuosity approximation
Dr = (1-f).*D;

% axonal stick compartment
Sa = sqrt(pi./(4*(b.*D))) .* erf(sqrt(b.*D)) .* exp(-te.*R2a);
% extraceullular axi-symmetric compartment
Se = sqrt(pi./(4.*(D - Dr).*b)) .* exp(-b.*Dr) .* erf(sqrt(b .*(D - Dr))) .* exp(-te.*R2e);
% combining signal
s = f.*Sa + (1-f).*Se;
% s = f.*Sa + (1-f-fcsf).*Se + fcsf.*Scsf;    % 20240614: matching original code

end