classdef SmokeFit_NEXITest < matlab.unittest.TestCase
    % Tier-2 smoke test for gpuNEXI: fit a tiny synthetic multi-shell DWI
    % dataset and check the fit runs and produces finite output. Modeled
    % on NEXI/demo_gpuNEXI_NoisePropagation.m, shrunk from 1e3 voxels /
    % 15 b-values down to a handful of each for speed.
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
            bval_sorted   = [2.3, 3.5, 4.8, 6.5, 2.3, 3.5];   % ms/um^2
            BDELTA_sorted = [13, 13, 13, 13, 21, 21];         % ms

            pars.fa = 0.1 + 0.7 * rand(1, Nsample);
            pars.Da = 1.5 + 1.5 * rand(1, Nsample);
            pars.De = 0.5 + 1.0 * rand(1, Nsample);
            tex     = 2   + 48  * rand(1, Nsample);
            pars.ra = (1 - pars.fa) ./ tex;
            pars.p2 = 0.05 + 0.45 * rand(1, Nsample);

            lmax   = 2;
            objGPU = gpuNEXI(bval_sorted, BDELTA_sorted);
            s      = objGPU.FWD(pars, lmax);
            noise  = 1 / 50;
            y      = s + noise * randn(size(s));
            y      = permute(y, [3 2 4 1]);
            mask   = true(size(y, 1:3));

            fitting              = [];
            fitting.solver       = 'askadam';
            fitting              = objGPU.check_set_default(fitting);
            fitting.lmax         = lmax;
            fitting.start        = 'likelihood';
            fitting.iteration    = 50;
            extraData            = [];

            out = objGPU.estimate(y, mask, extraData, fitting);

            testCase.verifyTrue(isfield(out, 'final'));
            for name = {'fa', 'Da', 'De', 'ra', 'p2'}
                testCase.verifyTrue(isfield(out.final, name{1}));
                testCase.verifyTrue(all(isfinite(out.final.(name{1})(:))));
            end
        end
    end
end
