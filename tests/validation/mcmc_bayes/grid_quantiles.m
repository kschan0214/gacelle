function [xq, fq, sd] = grid_quantiles(x, p, qs)
% GRID_QUANTILES Quantiles and density of a 1D marginal given on grid nodes.
%
%   [xq, fq, sd] = grid_quantiles(x, p, qs)
%
% x   : node positions (increasing, uniform or not)
% p   : unnormalised density at the nodes
% qs  : probabilities
% xq  : quantiles (trapezoid CDF, linear interpolation between nodes)
% fq  : normalised density at xq (linear interpolation)
% sd  : marginal standard deviation (trapezoid)
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%
x = double(x(:)); p = double(p(:));
cdf = [0; cumsum(0.5*(p(1:end-1)+p(2:end)).*diff(x))];
Z   = cdf(end);
cdf = cdf/Z; p = p/Z;
xq  = zeros(size(qs)); fq = zeros(size(qs));
for k = 1:numel(qs)
    j = find(cdf >= qs(k), 1);
    if j == 1; xq(k) = x(1); fq(k) = p(1); continue; end
    % inside cell [j-1, j] the density is linear, so the CDF is quadratic: solve exactly
    h  = x(j)-x(j-1); p0 = p(j-1); p1 = p(j); r = qs(k) - cdf(j-1);
    a2 = (p1-p0)/(2*h);
    if abs(a2) < 1e-14*max(p0,p1)/h
        dx = r/p0;
    else
        dx = (-p0 + sqrt(max(p0^2 + 4*a2*r, 0)))/(2*a2);
    end
    xq(k) = x(j-1) + dx;
    fq(k) = p0 + (p1-p0)*dx/h;
end
m1 = trapz(x, x.*p); sd = sqrt(max(trapz(x, (x-m1).^2.*p), 0));
end
