function [Sreal,Simag] = compute_gremwi_signal(S0MW,S0IW,S0EW,r2sMW,r2sIW,r2sEW,freqMW,freqIW,freqEW,freqBKG,pini,decayEW,ff,TE,B0,gyro,S0IEW_phase)

% if nargin < 17 || isempty(S0IEW_phase)
%     S0IEW_phase = 0;
% end

Sreal = (   S0MW .* exp(-TE .* r2sMW) .* cos(TE .* 2.*pi.*(freqMW+freqBKG).*B0.*gyro + pini) + ...
            S0IW .* exp(-TE .* r2sIW) .* cos(TE .* 2.*pi.*(freqIW+freqBKG).*B0.*gyro + pini  + S0IEW_phase) + ...
            S0EW .* exp(-TE .* r2sEW) .* cos(TE .* 2.*pi.*(freqEW+freqBKG).*B0.*gyro + pini  + S0IEW_phase) .* exp(-decayEW) ).*ff;

Simag = (   S0MW .* exp(-TE .* r2sMW) .* sin(TE .* 2.*pi.*(freqMW+freqBKG).*B0.*gyro + pini) + ...
            S0IW .* exp(-TE .* r2sIW) .* sin(TE .* 2.*pi.*(freqIW+freqBKG).*B0.*gyro + pini  + S0IEW_phase) + ...
            S0EW .* exp(-TE .* r2sEW) .* sin(TE .* 2.*pi.*(freqEW+freqBKG).*B0.*gyro + pini  + S0IEW_phase) .* exp(-decayEW) ).*ff;

end
