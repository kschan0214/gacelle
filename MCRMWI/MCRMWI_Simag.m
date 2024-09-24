
function Simag = MCRMWI_Simag(S0MW,S0IEW,iwf,r2sMW,r2sIW,r2sEW,freqMW,freqIW,freqEW,freqBKG,pini,S0IEW_phase,decayEW,TE,B0,gyro)

Simag = (   S0MW            .* exp(-TE .* r2sMW) .* sin(TE .* 2.*pi.*(freqMW+freqBKG).*B0.*gyro + pini) + ...               % MW
            S0IEW.*iwf      .* exp(-TE .* r2sIW) .* sin(TE .* 2.*pi.*(freqIW+freqBKG).*B0.*gyro + pini + S0IEW_phase) + ...    % IW
            S0IEW.*(1-iwf)  .* exp(-TE .* r2sEW) .* sin(TE .* 2.*pi.*(freqEW+freqBKG).*B0.*gyro + pini + S0IEW_phase) .* exp(-decayEW) );

end