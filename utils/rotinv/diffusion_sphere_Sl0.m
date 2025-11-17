%% S = diffusion_sphere_Sl0(b,D)
%
% Input
% --------------
% b             : b-value
% D             : diffusivity
%
% Output
% --------------
% signal        : signal decay due to diffusion
%
% Description:
%
% Kwok-shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 13 November 2025
% Date modified:
%
%
function S = diffusion_sphere_Sl0(b,D)

    S = exp(-b.*D) ;

end