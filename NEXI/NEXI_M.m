%% s = NEXI_M(x, fa, Da, d2, r1, r2)
%
% Description: Support function for gpuNEXImcmc to be able to use arrayfunc
%              Compute the combined dMRI signal ffrom all compartments
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 17 June 2024
% Date modified: 
%
%
function M = NEXI_M(x, fa, Da, d2, r1, r2)
    d1 = Da.*x.^2;
    l1 = (r1+r2+d1+d2)/2;
    l2 = sqrt( (r1-r2+d1-d2).^2 + 4*r1.*r2 )/2; 
    lm = l1-l2;
    Pp = (fa.*d1 + (1-fa).*d2 - lm)./(l2*2);
    M  = Pp.*exp(-(l1+l2)) + (1-Pp).*exp(-lm); 
end