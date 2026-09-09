classdef SmokeFit_AxCaliberSMTTest < matlab.unittest.TestCase
    % Tier-2 smoke test for gpuAxCaliberSMT: fit a tiny synthetic
    % multi-shell DWI dataset and check the fit runs and produces finite
    % output. Modeled on
    % AxCaliberSMT/demo_gpuAxCaliberSMT_NoisePropagation.m, shrunk from
    % 1e3 voxels / 16 b-values down to a handful of each for speed.
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

            Nsample       = 8;
            D0            = 1.7; Da_fixed = 1.7; DeL_fixed = 1.7; Dcsf = 3;
            bval_sorted   = [0.05, 0.35, 0.80, 1.5, 2.401, 3.45];
            ldelta_sorted = ones(size(bval_sorted)) * 6;
            BDELTA_sorted = [13, 13, 13, 13, 13, 13];

            pars.a    = 0.5 + 5.5 * rand(1, Nsample);
            pars.f    = 0.3 + 0.7 * rand(1, Nsample);
            pars.fcsf = 0.3 * rand(1, Nsample);
            pars.DeR  = 0.5 + 1.0 * rand(1, Nsample);

            model  = 'VanGelderen';
            objGPU = gpuAxCaliberSMT(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);
            s      = objGPU.FWD(pars, model);
            noise  = 1 / 100;
            y      = s + noise * randn(size(s));
            y      = permute(y, [2 3 4 1]);
            mask   = true(size(y, 1:3));

            fitting              = [];
            fitting.solver       = 'askadam';
            fitting              = objGPU.check_set_default(fitting);
            fitting.start        = 'likelihood';
            fitting.iteration    = 50;
            extraData            = [];

            out = objGPU.estimate(y, mask, extraData, fitting);

            testCase.verifyTrue(isfield(out, 'final'));
            for name = {'a', 'f', 'fcsf', 'DeR'}
                testCase.verifyTrue(isfield(out.final, name{1}));
                testCase.verifyTrue(all(isfinite(out.final.(name{1})(:))));
            end
        end
    end
end
