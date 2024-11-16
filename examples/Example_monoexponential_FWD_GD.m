%% S = Example_monoexponential_FWD_askadam( pars, t)
%
% Input
% --------------
% pars          : input model parameter structure (This is ALWAYS the first input variable)
% t             : [1xNt] sampling time (could be any extra input)
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
% Date last modified: 7 Nov 2024
%
%
function S = Example_monoexponential_FWD_GD( pars, t)
    
% columnised t
t = t(:);

% S0 and R2star here are [1xNvoxel] arrays
S0      = pars.S0;
R2star  = pars.R2star;

% compute S, as [NtxNvoxel] matrix
S = S0 .* exp(-t.*R2star);

end