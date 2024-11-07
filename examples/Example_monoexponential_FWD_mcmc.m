%% S = Example_monoexponential_FWD_mcmc( pars, t)
%
% Input
% --------------
% pars          : input model parameter structure (This is ALWAYS the first input variable)
% t             : [1xNt] sampling time
%
% Output
% --------------
% S             : monoexponential decay signal, [NtxNvoxel] matrix
%
% Description: example forward function for askadam solver
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 24 September 2024
% Date last modified:
%
%
function S = Example_monoexponential_FWD_mcmc( pars, t)
    
% columnised t
t = t(:);

% make sure the first dimension is 1
if size(pars.S0,1) ~= 1
    S0      = shiftdim(pars.S0,-1);
    R2star  = shiftdim(pars.R2star,-1);
else
    S0      = pars.S0;
    R2star  = pars.R2star;
end

% compute S, as [NtxNvoxel] matrix
S = S0 .* exp(-t.*R2star);

end