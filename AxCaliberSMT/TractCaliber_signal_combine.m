%% s = TractCaliber_signal_combine(C, f, fcsf, DeR, b,Scsf)
%
% Input
% --------------
%
% Output
% --------------
%
% Description: Support function for gpuTractCalibermcmc to be able to use arrayfunc
%              Compute the combined dMRI signal from all compartments
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 28 March 2024
% Date modified:
%
%
function s = TractCaliber_signal_combine(C, f, fcsf, DeR, b,Scsf)

% S = exp(-C);
% Sa = S;
% % 2. Extra-cellular signal
% Se = exp(-this.b.*DeR);
% % Combined signal
% s = (1-fcsf).*(f.*Sa + (1-f).*Se) + fcsf.*this.Scsf;

% Sa = exp(-C);
% Se = exp(-b.*DeR);

s = (1-fcsf).*(f.*exp(-C) + (1-f).*exp(-b.*DeR)) + fcsf.*Scsf;

end