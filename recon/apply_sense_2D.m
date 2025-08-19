% function Sc = apply_sense_2D( pars, coilSensMap, mask, iskspace,mask_kspace)
% 
% Input
% ---------------
% pars          : structure contains all estimation parameters
% coilSensMap   : coil sensitivity maps
% mask_kspace   : kspace undersampling mask
% isKspace      : convert coil images to kspace
% 
% Output
% ---------------
% Sc            : coil signal (either image or kspace)
%
% Description: 
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 25 July 2025
% Date modified:
%
function Sc = apply_sense_2D( pars, mask,coilSensMap, iskspace, mask_kspace)

S   = pars.S;
if any(size(S,1:3) ~= size(mask,1:3))  % if the size doesn't match then the input is masked
    S = utils.reshape_GD2ND(S,mask);  % restore original shape
end

% coil images
Sc = S .* coilSensMap;

if iskspace
    % 2D fft and apply kspace mask
    Sc = gacelleFFT.fft2(Sc) .* mask_kspace;
    % separate real and imaginary parts
    Sc = cat(4,real(Sc),imag(Sc));
end

end
