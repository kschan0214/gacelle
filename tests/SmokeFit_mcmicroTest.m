classdef SmokeFit_mcmicroTest < matlab.unittest.TestCase
    % Tier-2 smoke test for gpumcmicro: fit a tiny synthetic spherical-mean
    % DWI dataset and check the fit runs and produces finite output.
    % Modeled on mcmicro/demo_noise_propagation.m, shrunk from 1e3 voxels
    % down to a handful for speed. Uses the (data, mask, extraData,
    % fitting) argument order (fixed to match every other model class).
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

            seed = 1; rng(seed); gpurng(seed);

            Nsample     = 8;
            bval_sorted = [0, 0.5, 1, 2];

            pars.f = 0.1 + 0.7 * rand(1, Nsample);
            pars.D = 1.5 + 1.5 * rand(1, Nsample);

            objGPU = gpumcmicro(bval_sorted);
            s      = objGPU.FWD(pars);
            noise  = 1 / 40;
            y      = s + noise * randn(size(s));
            y      = permute(y, [3 2 4 1]);
            mask   = true(size(y, 1:3));

            fitting              = [];
            fitting.solver       = 'askadam';
            fitting              = objGPU.check_set_default(fitting);
            fitting.iteration    = 50;

            out = objGPU.estimate(y, mask, [], fitting);

            testCase.verifyTrue(isfield(out, 'final'));
            testCase.verifyTrue(isfield(out.final, 'f'));
            testCase.verifyTrue(isfield(out.final, 'D'));
            testCase.verifyTrue(all(isfinite(out.final.f(:))));
            testCase.verifyTrue(all(isfinite(out.final.D(:))));
        end
    end
end
