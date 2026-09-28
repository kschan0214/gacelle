function S = diffusion_relaxation_SMT_stick(b,te,C,r,Da, R2a, k2a)

    S = sqrt(pi./(4*(b.*Da - C))) .* exp(-C) .* erf(sqrt(b.*Da - C)) .* exp(-te.*( R2a + k2a./r));

end