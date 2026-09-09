classdef SmokeFit_SANDITest < matlab.unittest.TestCase
    % Tier-2 smoke test for gpuSANDI: fit a tiny synthetic multi-shell DWI
    % dataset and check the fit runs and produces finite output. Modeled
    % on SANDI/demo_gpuSANDI_NoisePropagation.m, shrunk from 1e3 voxels
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

            Nsample       = 8;
            Ds            = 3;
            pulseType     = 'wide';
            bval_sorted   = [0.05, 0.3, 0.8, 1.5, 2.3, 3.5];
            BDELTA_sorted = 13 * ones(size(bval_sorted));
            ldelta_sorted = 6  * ones(size(bval_sorted));

            pars.f  = 0.1 + 0.7 * rand(1, Nsample);
            pars.Da = 1.5 + 1.5 * rand(1, Nsample);
            pars.De = 0.5 + 1.0 * rand(1, Nsample);
            pars.Rs = 5   + 5   * rand(1, Nsample);
            pars.fs = 0.1 + 0.7 * rand(1, Nsample);

            objGPU = gpuSANDI(bval_sorted, ldelta_sorted, BDELTA_sorted, Ds);
            s      = objGPU.FWD(pars, pulseType);
            noise  = 1 / 50;
            y      = s + noise * randn(size(s));
            y      = permute(y, [3 2 4 1]);
            mask   = true(size(y, 1:3));

            fitting              = [];
            fitting.solver       = 'askadam';
            fitting              = objGPU.check_set_default(fitting);
            fitting.start        = 'likelihood';
            fitting.iteration    = 50;
            extraData            = [];

            out = objGPU.estimate(y, mask, extraData, fitting);

            testCase.verifyTrue(isfield(out, 'final'));
            for name = {'f', 'Da', 'De', 'Rs', 'fs'}
                testCase.verifyTrue(isfield(out.final, name{1}));
                testCase.verifyTrue(all(isfinite(out.final.(name{1})(:))));
            end
        end
    end
end
