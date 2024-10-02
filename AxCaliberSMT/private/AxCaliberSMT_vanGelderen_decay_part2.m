%% s = AxCaliberSMT_vanGelderen_decay_part2(s,r,D0,g)
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
function s = AxCaliberSMT_vanGelderen_decay_part2(s,r,D0,g)

s = s.*D0.*g.^2.*(r.^2/D0).^3;

end