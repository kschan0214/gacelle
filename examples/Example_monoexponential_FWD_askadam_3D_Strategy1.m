%% S = Example_monoexponential_FWD_askadam( pars, t, mask)
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
function S = Example_monoexponential_FWD_askadam_3D_Strategy1( pars, t, mask)
    
% In this example we put the time in the 4th dimension
t = reshape(t(:),1,1,1,numel(t));

% S0 and R2tar are N-D array (1<=N<=3)
S0      = pars.S0;
R2star  = pars.R2star;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% to maximise the GPU memory efficiency, pars.S0 and pars.R2star are masked inside askAdam.m by default, therefore they have size of [1*Nvoxel]
% therefore, we need to store the 1D arrays to their unmasked shape
% the utility function 'reshape_GD2ND' can convert the masked 1D array back to its original ND shape given the signal mask
if any(size(S0,1:3) ~= size(mask,1:3))  % if the size doesn't match then the input is masked
    S0 = utils.reshape_GD2ND(S0,mask);  % restore original shape
end
if any(size(R2star,1:3) ~= size(mask,1:3))
    R2star = utils.reshape_GD2ND(R2star,mask);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute S, as [Nx*Ny*Nz*Nt] matrix
% since ndims(S)>2, masking will be automatically applied in askAdam.m
S = S0 .* exp(-t.*R2star);

end