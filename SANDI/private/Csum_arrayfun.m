%% C = Csum_arrayfun(delta, BDELTA, r, D, bm2)
%
% Input
% --------------
% delta         : diffusion pulsed-gradient duration [ms]
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
function C = Csum_arrayfun(delta, BDELTA, td, bm2)

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

end