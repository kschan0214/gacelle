classdef SmokeFit_R2starMappingTest < matlab.unittest.TestCase
    % Tier-2 smoke test for gpuR2starMapping: fit a tiny synthetic
    % mono-exponential R2* decay dataset and check the fit runs and
    % produces finite output. Modeled on
    % R2star/demo_gpuR2starMapping_NoisePropagation.m and
    % examples/Example_monoexponential_automem.m, shrunk from 1e3 voxels
    % / 1e4 iterations down to a handful of each for speed.
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

            Nsample = 8;
            te      = linspace(0, 40e-3, 6); % s

            pars.M0     = 1   + 0.1 * rand(1, Nsample);
            pars.R2star = 30  + 5   * rand(1, Nsample);

            objGPU = gpuR2starMapping(te);
            s      = objGPU.FWD(pars);
            noise  = mean(pars.M0) / 50;
            y      = s + noise * randn(size(s));
            y      = permute(y, [2 3 4 1]); % [x,y,z,TE]
            mask   = true(size(y, 1:3));

            fitting              = [];
            fitting.solver       = 'askadam';
            fitting              = objGPU.check_set_default(fitting);
            fitting.start        = 'default';
            fitting.iteration    = 50;

            out = objGPU.estimate(y, mask, fitting);

            testCase.verifyTrue(isfield(out, 'final'));
            testCase.verifyTrue(isfield(out.final, 'M0'));
            testCase.verifyTrue(isfield(out.final, 'R2star'));
            testCase.verifyEqual(size(out.final.M0), size(y, 1:3));
            testCase.verifyTrue(all(isfinite(out.final.M0(:))));
            testCase.verifyTrue(all(isfinite(out.final.R2star(:))));
        end
    end
end
