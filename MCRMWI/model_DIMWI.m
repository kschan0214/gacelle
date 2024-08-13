%% [Sreal,Simag] = model_DIMWI(te, B0, gyro, S0MW, S0IW, S0EW, t2sMW, t2sIW, t2sEW, decayEW, freqMWBKG, freqIWBKG, freqEWBKG, pini)
%
% Input
% --------------
%
% Output
% --------------
%
% Description: Support function for gpuDIMWImcmc to be able to use arrayfunc
%              Compute the combined GRE signal from all compartments
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 17 July 2024
% Date modified: 
%
%
function [Sreal,Simag] = model_DIMWI(te, B0, gyro, S0MW, S0IW, S0EW, r2sMW, r2sIW, r2sEW, decayEW, freqMWBKG, freqIWBKG, freqEWBKG, pini)
            
Sreal = S0MW .* exp(-te .* r2sMW) .* cos(te .* 2*pi*(freqMWBKG)*B0*gyro + pini) + ...
        S0IW .* exp(-te .* r2sIW) .* cos(te .* 2*pi*(freqIWBKG)*B0*gyro + pini) + ...
        S0EW .* exp(-te .* r2sEW) .* cos(te .* 2*pi*(freqEWBKG)*B0*gyro + pini) .* exp(decayEW);

Simag = S0MW .* exp(-te .* r2sMW) .* sin(te .* 2*pi*(freqMWBKG)*B0*gyro + pini) + ...
        S0IW .* exp(-te .* r2sIW) .* sin(te .* 2*pi*(freqIWBKG)*B0*gyro + pini) + ...
        S0EW .* exp(-te .* r2sEW) .* sin(te .* 2*pi*(freqEWBKG)*B0*gyro + pini) .* exp(decayEW);

end