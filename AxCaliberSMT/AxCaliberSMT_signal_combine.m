%% s = AxCaliberSMT_signal_combine(b,Da,C,DeL,DeR,f,fcsf,Scsf)
%
% Input
% --------------
%
% Output
% --------------
%
% Description: Support function for gpuAxCaliberSMTmcmc to be able to use arrayfunc
%              Compute the combined dMRI signal ffrom all compartments
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 22 March 2024
% Date modified: 14 June 2024
%
%
function s = AxCaliberSMT_signal_combine(C, f, fcsf, DeR, b,Da,DeL,Scsf)

Sa = sqrt(pi./(4*(b*Da - C))) .* exp(-C) .* erf(sqrt(b*Da - C));
Se = sqrt(pi./(4.*(DeL - DeR).*b)) .* exp(-b.*DeR) .* erf(sqrt(b .*(DeL - DeR)));

% s = (1-fcsf).*(f.*Sa + (1-f).*Se) + fcsf.*Scsf;
s = f.*Sa + (1-f-fcsf).*Se + fcsf.*Scsf;    % 20240614: matching original code

end