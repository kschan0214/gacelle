classdef SmokeFit_IVIMTest < matlab.unittest.TestCase
    % Tier-2 smoke test for gpuIVIM: fit a tiny synthetic bi-exponential IVIM dataset with
    % askadam and mcmc (MH and ensemble) and check the fits run and produce finite output
    % inside the bounds; also check the b-value unit guard and that out.signalScale restores S0
    % in data units.
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
        function testAskadamFitRunsAndIsFinite(testCase)
            gacelletest.assumeGPU(testCase);
            [y, mask, b, T] = SmokeFit_IVIMTest.data();
            out = gpuIVIM(b).estimate(y, mask, [], struct('solver','askadam','iteration',50));
            SmokeFit_IVIMTest.verifyMaps(testCase, out, 'final', mask);
            % S0 in data units within 10% of the truth
            S0 = out.final.S0 .* out.signalScale;
            testCase.verifyLessThan(max(abs(S0(:) ./ T.S0(:) - 1)), 0.1);
        end

        function testMcmcFitRunsAndIsFinite(testCase)
            gacelletest.assumeGPU(testCase);
            [y, mask, b] = SmokeFit_IVIMTest.data();
            f   = struct('solver','mcmc','algorithm','MH','iteration',200,'burnin',0.5,'thinning',5,'metric',{{'mean','std'}});
            out = gpuIVIM(b).estimate(y, mask, [], f);
            SmokeFit_IVIMTest.verifyMaps(testCase, out, 'mean', mask);
            f   = struct('solver','mcmc','algorithm','ensemble','Nwalker',8,'StepSize',2,'iteration',100, ...
                         'burnin',0.5,'thinning',5,'metric',{{'mean','std'}});
            out = gpuIVIM(b).estimate(y, mask, [], f);
            SmokeFit_IVIMTest.verifyMaps(testCase, out, 'mean', mask);
        end

        function testBvalueUnitGuard(testCase)
            testCase.verifyError(@() gpuIVIM([0 50 100 200 500 800]), 'gpuIVIM:bval');   % s/mm2
            testCase.verifyError(@() gpuIVIM([0 0 0.5]), 'gpuIVIM:bval');                % < 3 distinct b
        end
    end

    methods (Static)
        function [y, mask, b, T] = data()
            seed = 1; rng(seed); gpurng(seed);
            sz   = [3 3 1];
            b    = [0 10 20 40 80 150 200 400 600 800] / 1000;          % ms/um2
            T.S0 = 500 * (0.8 + 0.4*rand(sz));
            T.f  = 0.05 + 0.15*rand(sz); T.D = 0.6 + 0.6*rand(sz); T.Dstar = 10 + 30*rand(sz);
            bb   = reshape(b, 1, 1, 1, []);
            y    = T.S0 .* (T.f.*exp(-bb.*T.Dstar) + (1-T.f).*exp(-bb.*T.D));
            y    = y + 500/100 * randn(size(y));
            mask = true(sz);
        end

        function verifyMaps(testCase, out, metric, mask)
            o  = gpuIVIM([0 0.1 0.5]);
            for k = 1:4
                p = o.modelParams{k};
                x = out.(metric).(p);
                testCase.verifyEqual(size(x, 1:3), size(mask, 1:3), p);
                testCase.verifyTrue(all(isfinite(x(:))), p);
                testCase.verifyTrue(all(x(mask) >= o.lb(k) & x(mask) <= o.ub(k)), p);
            end
        end
    end
end
