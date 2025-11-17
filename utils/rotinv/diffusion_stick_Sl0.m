%% S = diffusion_stick_Sl0(b,Da)
%
% Input
% --------------
% b             : b-value
% D             : longitudinal diffusivity
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
function S = diffusion_stick_Sl0(b,D)
    epsilon = 1e-8;

    bD = max(b.*D,epsilon);

    S = sqrt(pi./(4*(bD))) .* erf(sqrt(bD)) ;

end