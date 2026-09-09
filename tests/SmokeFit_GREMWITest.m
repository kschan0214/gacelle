classdef SmokeFit_GREMWITest < matlab.unittest.TestCase
    % Tier-2 smoke test for gpuGREMWI: fit a tiny synthetic complex
    % multi-echo GRE dataset and check the fit runs and produces finite
    % output. Modeled on MCRMWI/demo_gpuGREMWI_noisePropagation.m, shrunk
    % from 1e3 voxels / 15 echoes down to a handful of each for speed.
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
            Nt      = 6;
            TE1     = 1.5e-3; tr = 45e-3;
            t       = linspace(TE1, tr - 3e-3, Nt);
            B0      = 3;

            pars.S0      = 0.5   + 1.5  * rand(1, Nsample);
            pars.MWF     = 1e-8  + 0.25 * rand(1, Nsample);
            pars.IWF     = 0.2   + 0.6  * rand(1, Nsample);
            pars.R2sMW   = 75    + 75   * rand(1, Nsample);
            pars.R2sIW   = min(8 + 57 * rand(1, Nsample), 50);
            pars.R2sEW   = min(pars.R2sIW + 10 * rand(1, Nsample), 50);
            pars.freqMW  = (0 + 20 * rand(1, Nsample)) / B0 / gpuGREMWI.gyro;
            pars.freqIW  = (-10 + 10 * rand(1, Nsample)) / B0 / gpuGREMWI.gyro;
            pars.dfreqBKG = -0.05 + 0.1 * rand(1, Nsample);
            pars.dpini    = -1 + 2 * rand(1, Nsample);

            fitting = [];
            fitting.DIMWI.isFitIWF    = true;
            fitting.DIMWI.isFitFreqMW = true;
            fitting.DIMWI.isFitFreqIW = true;
            fitting.DIMWI.isFitR2sEW  = true;
            fitting.isComplex = true;

            spatialSize = size(pars.S0, 1:3); % [1, Nsample, 1]
            extraData.freqBKG = zeros(spatialSize);
            extraData.pini    = zeros(spatialSize);
            extraData.ff      = ones(spatialSize);
            extraData.theta   = zeros(spatialSize);

            objGPU = gpuGREMWI(t);
            s      = (objGPU.FWD(pars, fitting, extraData)).';
            s      = permute(reshape(s, [Nsample, Nt, 2]), [1 4 5 2 3]);
            mask   = true(size(s, 1:3));

            noise = (pars.S0.') / 150;
            y     = s + noise .* randn(size(s));
            y     = y(:, :, :, :, 1) + 1i * y(:, :, :, :, 2);

            fitting              = [];
            fitting.solver       = 'askadam';
            fitting              = objGPU.check_set_default(fitting, y);
            fitting.start        = 'prior';
            fitting.iteration    = 50;

            out = objGPU.estimate(y, mask, extraData, fitting);

            testCase.verifyTrue(isfield(out, 'final'));
            for name = {'S0', 'MWF', 'IWF', 'R2sMW', 'R2sIW', 'R2sEW'}
                testCase.verifyTrue(isfield(out.final, name{1}));
                testCase.verifyTrue(all(isfinite(out.final.(name{1})(:))));
            end
        end
    end
end
