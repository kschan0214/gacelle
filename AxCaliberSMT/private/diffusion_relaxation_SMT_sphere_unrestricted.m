% unrestricted sphere
function S = diffusion_relaxation_SMT_sphere_unrestricted(b,te,D,R2)

    S = exp(-b.*D) .*exp(-te.*R2);

end