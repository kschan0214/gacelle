%% S = Example_monoexponential_FWD_askadam( pars, mask, t)
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
function S = Example_monoexponential_FWD_askadam_3D_Strategy2( pars, t, mask)
    
% In this example we put the time in the 4th dimension
t = reshape(t(:),1,1,1,numel(t));

% S0 and R2tar are N-D array (1<=N<=3)
S0      = pars.S0;
R2star  = pars.R2star;

% compute S, as [Nx*Ny*Nz*Nt] matrix
S = S0 .* exp(-t.*R2star);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% to maximise the GPU memory efficiency, pars.S0 and pars.R2star are masked inside askAdam.m by default, therefore they have size of [1*Nvoxel]
% that means the size of S will be [1*Nvoxel*1*Nt] during the askAdam optimisation loops
% since S only contains the masked voxel, we need to convert S into a 2D array to avoid additional masking step in askAdam.m
% the utility function 'reshape_ND2GD' can convert any N-D array (N>=4) into 2D array ([Nmeas*Nvoxel]) compatible for askAdam.m 
% compare to Strategy 1 this is a more memory efficient way since fewer total voxels are involved but less intuitive
if any(size(S0,1:3) ~= size(mask,1:3))  % if the size doesn't match then the input is masked
    S = utils.reshape_ND2GD(S,[]);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end