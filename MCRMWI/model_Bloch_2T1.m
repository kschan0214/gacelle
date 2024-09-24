% Bloch non-exchanging 3-pool steady-state model
    function [S0MW, S0IEW] = model_Bloch_2T1(TR,M0MW,M0IEW,T1MW,T1IEW,true_famp)
    % s     : non-exchange steady-state signal, 1st dim: pool; 2nd dim: flip angles
    %
        S0MW    = M0MW  .* sin(true_famp) .* (1-exp(-TR./T1MW)) ./(1-cos(true_famp).*exp(-TR./T1MW));
        S0IEW   = M0IEW .* sin(true_famp) .* (1-exp(-TR./T1IEW))./(1-cos(true_famp).*exp(-TR./T1IEW));

    end