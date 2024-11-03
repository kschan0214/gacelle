%% S = Example_monoexponential_FWD_askadam( pars, mask, t)
%
% Input
% --------------
% pars          : input model parameter structure (do not modified or change position)
% mask          : mask  (do not modified or change position)
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
function S = Example_monoexponential_FWD_askadam( pars, t)
    
% columnised t
t = t(:);

% convert S0 and R2star into row vectors for matrix multiplication
S0      = pars.S0;
R2star  = pars.R2star;

% compute S, as [NtxNvoxel] matrix
S = S0 .* exp(-t.*R2star);

end