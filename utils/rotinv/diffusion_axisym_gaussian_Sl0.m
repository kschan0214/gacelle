%% S = diffusion_axisym_gaussian_Sl0(b,Da,Dr)
%
% Input
% --------------
% b             : b-value
% Da            : longitudinal diffusivity
% Dr            : radial diffusivity
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
function S = diffusion_axisym_gaussian_Sl0(b,Da,Dr)

    epsilon = 1e-8;

    bdDe = max(b.*(Da-Dr),epsilon);                    % avoid division by zeros and negative values for sqrt

    S = sqrt(pi./(4.*bdDe)) .* exp(-b.*Dr) .* erf(sqrt(bdDe)) ;

end