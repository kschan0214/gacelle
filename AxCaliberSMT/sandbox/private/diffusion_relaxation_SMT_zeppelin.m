% zeppelin
function S = diffusion_relaxation_SMT_zeppelin(b,te,Da,Dr,R2e)

    dDe = Da - Dr;              
    % dDe = max(dDe, gpuMEAxCaliberSMTmcmc.epsilon);    % avoid division by zeros and negative values for sqrt

    S = sqrt(pi./(4.*b.*dDe)) .* exp(-b.*Dr) .* erf(sqrt(b .*dDe)) .* exp(-te.*R2e);

end