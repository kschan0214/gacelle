classdef SmokeFit_MEAxCaliberSMTTest < matlab.unittest.TestCase
    % Tier-2 smoke test for gpuMEAxCaliberSMT (EXPERIMENTAL): fit a tiny synthetic two-TE
    % spherical-mean dataset with askadam and mcmc and check the fits run and give finite output
    % inside the bounds. Note the argument order estimate(data, mask, fitting, extraData).
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
            [y, mask, obj] = McmcBayesLegacyTest.meaxcaliberData();
            out = obj().estimate(y, mask, struct('solver','askadam','iteration',50,'start','default'), []);
            SmokeFit_MEAxCaliberSMTTest.verifyMaps(testCase, out.final, mask);
        end

        function testMcmcFitRunsAndIsFinite(testCase)
            gacelletest.assumeGPU(testCase);
            [y, mask, obj] = McmcBayesLegacyTest.meaxcaliberData();
            f   = struct('solver','mcmc','algorithm','MH','iteration',200,'burnin',0.5,'thinning',5, ...
                         'metric',{{'mean','std'}},'start','default');
            out = obj().estimate(y, mask, f, []);
            SmokeFit_MEAxCaliberSMTTest.verifyMaps(testCase, out.mean, mask);
        end
    end

    methods (Static)
        function verifyMaps(testCase, res, mask)
            o = gpuMEAxCaliberSMT([0 1], [6 6], [13 13], [0.051 0.051], [], []);
            for p = {'f','fcsf','DeR','r','R2e'}
                k = find(strcmp(o.modelParams, p{1}));
                x = res.(p{1});
                testCase.verifyEqual(size(x, 1:3), size(mask, 1:3), p{1});
                testCase.verifyTrue(all(isfinite(x(:))), p{1});
                testCase.verifyTrue(all(x(mask) >= o.lb(k) & x(mask) <= o.ub(k)), p{1});
            end
        end
    end
end
