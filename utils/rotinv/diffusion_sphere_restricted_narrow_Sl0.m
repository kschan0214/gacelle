%% S = diffusion_sphere_restricted_narrow_Sl0(delta, g, r, D)
%
% Input
% --------------
% delta         : diffusion pulsed-gradient duration [ms]
% g             : diffusion gradient, g = sqrt(b./delta.^2./(BDELTA-delta/3));
% r             : sphere radius [um]
% D             : intrinsic diffusivity [um2/ms]
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
function S = diffusion_sphere_restricted_narrow_Sl0(delta, g, r, D)
    epsilon = 1e-8;

    D = max(D,epsilon);

    C = (16/175)*g.^2.*delta.*r.^4./D;

    S = exp(-C);

end