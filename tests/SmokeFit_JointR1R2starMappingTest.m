classdef SmokeFit_JointR1R2starMappingTest < matlab.unittest.TestCase
    % Tier-2 smoke test for gpuJointR1R2starMapping: fit a tiny synthetic
    % multi-flip-angle, multi-echo GRE dataset and check the fit runs and
    % produces finite output. Modeled on
    % R1R2s/demo_gpuJointR1R2starMapping_NoisePropagation.m, shrunk from
    % 1e3 voxels / 15 echoes / 7 flip angles down to a handful of each
    % for speed.
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

            Nsample = 6;
            Nt      = 5;
            TE1     = 1.5e-3; tr = 45e-3;
            t       = linspace(TE1, tr - 3e-3, Nt);
            fa      = [5, 20, 50];
            Nfa     = numel(fa);

            pars.M0     = 0.5 + 1.5 * rand(1, Nsample);
            pars.R1     = 0.2 + 2.3 * rand(1, Nsample);
            pars.R2star = 5   + 55  * rand(1, Nsample);

            b1 = ones(size(pars.M0));
            extraData.b1 = b1;
            objGPU = gpuJointR1R2starMapping(t, tr, fa);
            s      = gather(objGPU.FWD(pars, extraData));
            s      = permute(reshape(s, [Nt, Nfa, Nsample]), [3 4 5 1 2]);
            mask   = true(size(s, 1:3));

            noise = (pars.M0.') / 100;
            y     = s + noise .* randn(size(s));

            extraData      = [];
            extraData.b1   = b1.';

            fitting              = [];
            fitting.solver       = 'askadam';
            fitting              = objGPU.check_set_default(fitting);
            fitting.start        = 'default';
            fitting.iteration    = 50;

            out = objGPU.estimate(y, mask, extraData, fitting);

            testCase.verifyTrue(isfield(out, 'final'));
            for name = {'M0', 'R1', 'R2star'}
                testCase.verifyTrue(isfield(out.final, name{1}));
                testCase.verifyTrue(all(isfinite(out.final.(name{1})(:))));
            end
        end
    end
end
