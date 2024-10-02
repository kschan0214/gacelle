%% s = AxCaliberSMT_vanGelderen_decay_part1(b,Da,C,DeL,DeR,f,fcsf,Scsf)
%
% Input
% --------------
%
% Output
% --------------
%
% Description: Support function for gpuAxCaliberSMTmcmc to be able to use arrayfunc
%              Compute the dMRI signal decay for an axon diameter using formula of van Gelderen
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 22 March 2024
% Date modified:
%
%
function s = AxCaliberSMT_vanGelderen_decay_part1(r,bm2,delta,Delta,D0)

s = (2/(bm2.^3.*(bm2-1))).*(-2 ...
                        + 2*(bm2.*(delta./(r.^2/D0))) ...
                        + 2*exp(-(bm2.*(delta./(r.^2/D0)))) ...
                        + 2*exp(-(bm2.*(Delta./(r.^2/D0)))) ...
                        - exp(-(bm2.*(Delta./(r.^2/D0)))-(bm2.*(delta./(r.^2/D0))))...
                        - exp(-(bm2.*(Delta./(r.^2/D0)))+(bm2.*(delta./(r.^2/D0)))));

end