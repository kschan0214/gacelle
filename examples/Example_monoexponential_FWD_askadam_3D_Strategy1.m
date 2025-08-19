%% S = Example_monoexponential_FWD_askadam_3D_Strategy1( pars, t, mask)
%
% Input
% --------------
% pars          : input model parameter structure (This is ALWAYS the first input variable)
% t             : [1xNt] sampling time (could be any extra input)
% mask          : 3D signal mask
%
% Output
% --------------
% S             : monoexponential decay signal, [Nx*Ny*Nz*Nvoxel] matrix
%
% Description: example forward function for askadam solver
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 14 Nov 2024
% Date last modified: 
%
function S = Example_monoexponential_FWD_askadam_3D_Strategy1( pars, t)
    
% In this example we put the time in the 4th dimension
t = reshape(t(:),1,1,1,numel(t));

% S0 and R2tar are N-D array (1<=N<=3)
S0      = pars.S0;
R2star  = pars.R2star;

% compute S, as [Nx*Ny*Nz*Nt] matrix
S = S0 .* exp(-t.*R2star);

end