classdef rician < handle
% functions related to rician noise
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 23 June 2025
% Date modified: 29 September 2026 (stable scaled-Bessel evaluation, no clamping; rician_mean static-call fix)
%
% First moment of the Rician distribution (magnitude of S + sigma*(n1 + i*n2), n1, n2 ~ N(0,1)):
%   E = sigma sqrt(pi/2) L_{1/2}(x),   x = -S^2/(2 sigma^2),
%   L_{1/2}(x) = e^(x/2) [(1-x) I0(-x/2) - x I1(-x/2)]
% For x <= 0, with t = -x/2 >= 0 and the exponentially scaled Bessel functions
% I0e(t) = e^(-t) I0(t), I1e(t) = e^(-t) I1(t):
%   L_{1/2}(x) = (1 - x) I0e(t) - x I1e(t)            (no overflow for any t)
% For t >= 1e4 the asymptotic form E = |S| (1 + 1/(8t)) = |S| + sigma^2/(2|S|) is used
% (relative error < 1e-8), so E -> |S| at high SNR, and sigma = 0 gives E = |S|.
%
% rician_mean / L12        : besseli with scaling (CPU or GPU; not for dlarray)
% rician_mean_gacelle /    : Abramowitz & Stegun 9.8.1-9.8.4 polynomials (|error| < 2.2e-7 of the
% L12_gacelle                scaled functions), written with elementwise operations and masks only,
%                            so that they work with single/double, gpuArray and dlarray (askadam
%                            autodiff); both branches are evaluated on clamped arguments so that
%                            neither the values nor the gradients contain Inf/NaN
% besseli_gacelle          : the previous trapezoidal approximation of the UNSCALED I_nu (kept for
%                            compatibility, no longer used here). It is accurate for moderate z but grows
%                            as e^z (overflow above z ~ 88 in single, ~ 709 in double), which is why the
%                            previous L12_gacelle clamped x >= -170: that made E = S (no Rician bias) for
%                            S/sigma > ~18.4, a relative error of -sigma^2/(2 S^2) (-1.25e-3 at S/sigma = 20)

    properties
    end

    properties (GetAccess = public, SetAccess = protected)

    end

    methods

        % % constructuor
        % function this = rician()
        %
        %
        % end


    end

    methods(Static)

        % for normal usage
        % first moment of Rician distribution
        function S_RM = rician_mean(S,sigma)
            t       = S.^2 ./ (4*sigma.^2);
            isHigh  = t >= 1e4;
            S_RM    = sigma .* sqrt(pi/2) .* rician.L12(-S.^2/2./sigma.^2);
            % asymptotic form at high SNR (also covers sigma = 0)
            S_HI    = abs(S) .* (1 + 1./(8*t));
            S_RM(isHigh) = S_HI(isHigh);
            S_RM    = max(S_RM, S);
        end

        % L_{1/2}(x) with exponentially scaled besseli (exact for any real x, no overflow for x <= 0)
        function y = L12(x)
            % besseli(nu,z,1) = exp(-|z|) besseli(nu,z), so e^(x/2) I_nu(-x/2) = e^(x/2 + |x|/2) besseli(nu,-x/2,1)
            y = exp(x/2 + abs(x)/2) .* ( (1-x).*besseli(0,-x/2,1) - x.*besseli(1,-x/2,1) );
        end

        % for gacelle
        % first moment of Rician distribution
        function S_RM = rician_mean_gacelle(S,sigma)
            t       = S.^2 ./ (4*sigma.^2);
            isHigh  = t >= 1e4;                         % NaN (S = sigma = 0) -> high branch, E = |S| = 0
            tMid    = min(t, 1e4);
            tHigh   = max(t, 1e4);
            E_mid   = sigma .* sqrt(pi/2) .* ((1 + 2*tMid).*rician.i0e_gacelle(tMid) + 2*tMid.*rician.i1e_gacelle(tMid));
            E_high  = abs(S) .* (1 + 1./(8*tHigh));
            S_RM    = E_mid.*(~isHigh) + E_high.*isHigh;
            S_RM    = max(S_RM, S);
        end

        function y = L12_gacelle(x)
            % x <= 0 (the Rician mean): t = -x/2, (1-x) I0e(t) - x I1e(t); for t >= 1e4 the
            % asymptotic sqrt(8t/pi) (1 + 1/(8t)); x > 0: e^x [(1-x) I0e(x/2) + x I1e(x/2)]
            isPos   = x > 0;
            t       = max(-x/2, 0);
            isHigh  = t >= 1e4;
            tMid    = min(t, 1e4);
            tHigh   = max(t, 1e4);
            yMid    = (1 + 2*tMid).*rician.i0e_gacelle(tMid) + 2*tMid.*rician.i1e_gacelle(tMid);
            yHigh   = sqrt(8*tHigh/pi) .* (1 + 1./(8*tHigh));
            xp      = max(x, 0);
            yPos    = exp(xp) .* ((1 - xp).*rician.i0e_gacelle(xp/2) + xp.*rician.i1e_gacelle(xp/2));
            y       = (yMid.*(~isHigh) + yHigh.*isHigh).*(~isPos) + yPos.*isPos;
        end

        % exponentially scaled modified Bessel functions I0e(z) = e^(-z) I0(z), I1e(z) = e^(-z) I1(z),
        % z >= 0, Abramowitz & Stegun 9.8.1-9.8.4; elementwise operations only (dlarray-compatible)
        function v = i0e_gacelle(z)
            isSmall = z < 3.75;
            zs      = min(z, 3.75);
            zl      = max(z, 3.75);
            t       = (zs/3.75).^2;
            vs      = (1 + t.*(3.5156229 + t.*(3.0899424 + t.*(1.2067492 + t.*(0.2659732 + t.*(0.0360768 + t.*0.0045813)))))) .* exp(-zs);
            u       = 3.75./zl;
            vl      = (0.39894228 + u.*(0.01328592 + u.*(0.00225319 + u.*(-0.00157565 + u.*(0.00916281 + u.*(-0.02057706 + ...
                       u.*(0.02635537 + u.*(-0.01647633 + u.*0.00392377)))))))) ./ sqrt(zl);
            v       = vs.*isSmall + vl.*(~isSmall);
        end

        function v = i1e_gacelle(z)
            isSmall = z < 3.75;
            zs      = min(z, 3.75);
            zl      = max(z, 3.75);
            t       = (zs/3.75).^2;
            vs      = zs.*(0.5 + t.*(0.87890594 + t.*(0.51498869 + t.*(0.15084934 + t.*(0.02658733 + t.*(0.00301532 + t.*0.00032411)))))) .* exp(-zs);
            u       = 3.75./zl;
            vl      = (0.39894228 + u.*(-0.03988024 + u.*(-0.00362018 + u.*(0.00163801 + u.*(-0.01031555 + u.*(0.02282967 + ...
                       u.*(-0.02895312 + u.*(0.01787654 - u.*0.00420059)))))))) ./ sqrt(zl);
            v       = vs.*isSmall + vl.*(~isSmall);
        end

        function I = besseli_gacelle(nu,z)
            Nx  = 34;    % NRMSE<0.05% for Nx=34
            x   = zeros([ones(1,ndims(z)), Nx]); x(:) = linspace(0,pi,Nx);
            I   = 1/pi * trapz(x(:),exp(z.*cos(x)).*cos(nu*x),ndims(x));
        end


    end

end
