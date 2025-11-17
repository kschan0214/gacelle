%% S = diffusion_sphere_restricted_narrow_Sl0(delta, g, r, D)
%
% Input
% --------------
% delta         : diffusion pulsed-gradient duration [ms]
% g             : diffusion gradient, g = sqrt(b./delta.^2./(BDELTA-delta/3));
% BDELTA        : diffusion time [ms]
% r             : sphere radius [um]
% D             : intrinsic diffusivity [um2/ms]
%
% Output
% --------------
% signal        : signal decay due to diffusion
%
% Description: arrayfun compatible function
%
% Kwok-shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 13 November 2025
% Date modified:
%
%
function S = diffusion_sphere_restricted_wide_Sl0_arrayfun(delta, g, BDELTA, r, D)

    bm2 = reshape([2.0816 5.9404 9.2058 12.4044 15.5792 ...
                   18.7426 21.8997 25.0528 28.2034 31.3521].^2, 1, 1, 1, []);

    if isgpuarray(r)
        bm2 = gpuArray(single(bm2));
    end

    td  = r.^2/D;
    C   = arrayfun(@Csum_arrayfun,delta, BDELTA, td, bm2);
    S   = arrayfun(@C_arrayfun,g, sum(C,4), D, td);

end