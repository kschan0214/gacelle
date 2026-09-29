classdef RicianUtilTest < matlab.unittest.TestCase
    % Tier-1 tests of utils/rician.m (first moment of the Rician distribution), plus a GPU check.
    %
    % Reference: E = int_0^inf y p(y) dy with the Rician density written with the scaled Bessel
    % function, p(y) = (y/s^2) exp(-(y-S)^2/(2 s^2)) besseli(0, y S/s^2, 1), integrated numerically
    % (integral, RelTol 1e-12, AbsTol 0, finite limits max(0, S - 40 s) to S + 50 s) in double.
    %
    % Tolerances (stated before running):
    %   S/sigma in {0, 0.1, 0.5, 1, 2, 3, 5, 10, 30, 100, 300, 1e3, 1e4} (low, moderate, high SNR):
    %     rician_mean (besseli)            : relative error <= 1e-8 (asymptotic branch above t = 1e4: < 1e-8)
    %     rician_mean_gacelle (A&S, double): relative error <= 1e-6
    %     rician_mean_gacelle (single, GPU): relative error <= 1e-5
    %   high SNR, S/sigma in [1e2, 1e6]: all finite, |E/S - 1 - sigma^2/(2 S^2)| <= 1e-6 (double);
    %     sigma = 0 gives E = S, S = 0 gives sigma sqrt(pi/2) (RelTol 1e-7)
    %   L12_gacelle vs L12 (besseli) for x in [-1e8, 0]: relative error <= 1e-6; x in (0, 20] (not used by
    %     the Rician mean; L_{1/2} changes sign there): |error| <= 1e-6 e^x
    %   dlarray (askadam autodiff): dlgradient of sum(rician_mean_gacelle) w.r.t. S and sigma finite and
    %     within 1e-5 (relative, + 1e-9 absolute) of central finite differences of rician_mean (double)
    %
    % Kwok-Shing Chan @ MGH

    methods (TestClassSetup)
        function addGacellePath(testCase)
            testsDir    = fileparts(mfilename('fullpath'));
            projectRoot = fileparts(testsDir);
            addpath(projectRoot);
            addpath_gacelle(projectRoot);
        end
    end

    methods (Test)
        function testRicianMeanVsIntegral(testCase)
            sigma = 0.05;
            snr   = [0 0.1 0.5 1 2 3 5 10 30 100 300 1e3 1e4];
            S     = snr * sigma;
            ref   = arrayfun(@(s) RicianUtilTest.meanIntegral(s, sigma), S);
            E1    = rician.rician_mean(S, sigma);
            E2    = rician.rician_mean_gacelle(S, sigma);
            testCase.verifyTrue(all(isfinite([E1 E2])));
            testCase.verifyLessThanOrEqual(max(abs(E1 - ref)./ref), 1e-8, 'rician_mean');
            testCase.verifyLessThanOrEqual(max(abs(E2 - ref)./ref), 1e-6, 'rician_mean_gacelle');
            % array sigma with scalar S, same values
            testCase.verifyEqual(rician.rician_mean_gacelle(S(8), sigma*[1 1]), E2(8)*[1 1], 'RelTol', 1e-12);
        end

        function testRicianMeanHighSnrAndLimits(testCase)
            sigma = 1;
            S     = logspace(2, 6, 200);
            for f = {@rician.rician_mean, @rician.rician_mean_gacelle}
                E = f{1}(S, sigma);
                testCase.verifyTrue(all(isfinite(E)), func2str(f{1}));
                testCase.verifyLessThanOrEqual(max(abs(E./S - 1 - sigma^2./(2*S.^2))), 1e-6, func2str(f{1}));
                testCase.verifyEqual(f{1}([0.3 2], 0), [0.3 2], func2str(f{1}));
                testCase.verifyEqual(f{1}(0, 0.2), 0.2*sqrt(pi/2), 'RelTol', 1e-7, func2str(f{1}));
            end
        end

        function testL12GacelleVsL12(testCase)
            x  = [-logspace(-8, 8, 400), 0];
            y1 = rician.L12(x); y2 = rician.L12_gacelle(x);
            testCase.verifyTrue(all(isfinite(y2)));
            testCase.verifyLessThanOrEqual(max(abs(y2 - y1)./abs(y1)), 1e-6);
            % x > 0 (not used by the Rician mean; L_{1/2} changes sign): error relative to the scale e^x
            x  = linspace(1e-3, 20, 50);
            y1 = rician.L12(x); y2 = rician.L12_gacelle(x);
            testCase.verifyLessThanOrEqual(max(abs(y2 - y1)./exp(x)), 1e-6);
        end

        function testDlarrayGradient(testCase)
            S   = [0.01 0.05 0.2 1 5];              % sigma = 0.05: S/sigma from 0.2 to 100
            sg  = 0.05;
            [E, gS, gSg] = dlfeval(@RicianUtilTest.meanAndGrad, dlarray(S), dlarray(sg));
            E = extractdata(E); gS = extractdata(gS); gSg = extractdata(gSg);
            testCase.verifyEqual(E, rician.rician_mean_gacelle(S, sg), 'RelTol', 1e-12);
            h   = 1e-6;
            fdS = (rician.rician_mean(S + h*S, sg) - rician.rician_mean(S - h*S, sg)) ./ (2*h*S);
            fdG = sum(rician.rician_mean(S, sg*(1+h)) - rician.rician_mean(S, sg*(1-h))) / (2*h*sg);
            testCase.verifyTrue(all(isfinite([gS gSg])));
            testCase.verifyLessThanOrEqual(max(abs(gS - fdS) - 1e-5*abs(fdS) - 1e-9), 0, 'dE/dS');
            testCase.verifyLessThanOrEqual(abs(gSg - fdG) - 1e-5*abs(fdG) - 1e-9, 0, 'dE/dsigma');
        end

        function testSingleGpu(testCase)
            gacelletest.assumeGPU(testCase);
            sigma = 0.05;
            snr   = [0 0.1 0.5 1 2 3 5 10 30 100 300 1e3 1e4 1e5];
            S     = gpuArray(single(snr * sigma));
            E     = double(gather(rician.rician_mean_gacelle(S, single(sigma))));
            Sd    = double(gather(S)); sd = double(single(sigma));
            ref   = arrayfun(@(s) RicianUtilTest.meanIntegral(s, sd), Sd);
            testCase.verifyTrue(all(isfinite(E)));
            testCase.verifyLessThanOrEqual(max(abs(E - ref)./ref), 1e-5);
        end
    end

    methods (Static)
        % E[|S + sigma (n1 + i n2)|] by numerical integration of the Rician density (double)
        function E = meanIntegral(S, sigma)
            p  = @(y) (y/sigma^2) .* exp(-(y - S).^2/(2*sigma^2)) .* besseli(0, y*S/sigma^2, 1);
            lo = max(0, S - 40*sigma); hi = S + 50*sigma;
            E  = integral(@(y) y.*p(y), lo, hi, 'RelTol', 1e-12, 'AbsTol', 0);
        end

        function [E, gS, gSg] = meanAndGrad(S, sg)
            E = rician.rician_mean_gacelle(S, sg);
            [gS, gSg] = dlgradient(sum(E), S, sg);
        end
    end
end
