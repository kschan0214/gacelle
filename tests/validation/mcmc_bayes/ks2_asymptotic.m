function [D, p] = ks2_asymptotic(x1, x2)
% KS2_ASYMPTOTIC Two-sample Kolmogorov-Smirnov test, asymptotic p-value.
% Self-contained replacement for kstest2 (Statistics toolbox).
%
%   [D, p] = ks2_asymptotic(x1, x2)
%
% D : sup |F1 - F2|
% p : asymptotic p-value with the Stephens (1970) small-sample correction,
%     lambda = (sqrt(ne) + 0.12 + 0.11/sqrt(ne)) * D, ne = n1*n2/(n1+n2),
%     p = 2 * sum_{j>=1} (-1)^(j-1) exp(-2 j^2 lambda^2)  (clipped to [0,1])
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%
x1 = double(x1(:)); x2 = double(x2(:));
n1 = numel(x1); n2 = numel(x2);
t  = [x1; x2].';
% right-continuous empirical CDFs at all pooled points (handles ties,
% e.g. repeated values from rejected MCMC moves)
F1 = mean(x1 <= t, 1);
F2 = mean(x2 <= t, 1);
D  = max(abs(F1 - F2));

ne     = n1*n2/(n1+n2);
lambda = (sqrt(ne) + 0.12 + 0.11/sqrt(ne)) * D;
j      = (1:101)';
p      = 2 * sum((-1).^(j-1) .* exp(-2 * j.^2 * lambda^2));
p      = min(max(p, 0), 1);
end
