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
% Description:
%
% Kwok-shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 13 November 2025
% Date modified:
%
%
function S = diffusion_sphere_restricted_wide_Sl0(delta, g, BDELTA, r, D)

    % bm2 = [2.0816    5.9404    9.2058    12.4044   15.5792 ...
    %        18.7426   21.8997   25.0528   28.2034   31.3521].^2;
    % bm2 = permute(bm2(:), [2 3 4 1]);
    bm2 = reshape([2.0816 5.9404 9.2058 12.4044 15.5792 ...
                   18.7426 21.8997 25.0528 28.2034 31.3521].^2, 1, 1, 1, []);

    if isgpuarray(r)
        bm2 = gpuArray(single(bm2));
    end

    td          = r.^2/D;
    bardelta    = delta./td ;
    barDelta    = BDELTA./td ;
    bm2bardelta = bm2.*bardelta;
    bm2barDelta = bm2.*barDelta;
    C           = (2./(bm2.^3.*(bm2-2))).*(-2 ...
                    + 2*bm2bardelta ...
                    + 2*exp(-bm2bardelta) ...
                    + 2*exp(-bm2barDelta) ...
                    - exp(-bm2barDelta-bm2bardelta)...
                    - exp(-bm2barDelta+bm2bardelta));
    C           = sum(C,4).*D.*g.^2.*td.^3;
    S = exp(-C);

end