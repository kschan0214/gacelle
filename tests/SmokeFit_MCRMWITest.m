classdef SmokeFit_MCRMWITest < matlab.unittest.TestCase
    % Tier-2 smoke test for gpuMCRMWI: fit a tiny synthetic complex,
    % variable-flip-angle, multi-echo GRE dataset and check the fit runs
    % and produces finite output. Modeled directly on
    % MCRMWI/demo_gpuMCRMWI_noisePropagation.m, shrunk from 1e3 voxels /
    % 15 echoes / 7 flip angles down to a handful of each for speed.
    %
    % This is the heaviest-setup model in the suite: it loads the
    % bundled pretrained EPG-X MLP weights
    % (MCRMWI/EPGXgen_net/MCRMWI_MLP_EPGX_RFphase50_T1M234_{magn,phase}.mat)
    % to forward-simulate ground truth via FWD(); estimate() itself
    % auto-loads those same weights internally, so it doesn't need them
    % passed in.
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

            Nsample = 4;
            Nt      = 5;
            TE1     = 1.5e-3; tr = 45e-3;
            te      = linspace(TE1, tr - 3e-3, Nt);
            fa      = [5, 20, 50];
            Nfa     = numel(fa);

            kappa_mw  = 0.36; kappa_iew = 0.86; % Jung, NI., water densities
            fixed_params.B0     = 3;
            fixed_params.rho_mw = kappa_mw / kappa_iew;
            fixed_params.E      = 0.02;
            fixed_params.x_i    = -0.1;
            fixed_params.x_a    = -0.1;
            fixed_params.B0dir  = [0; 0; 1];
            fixed_params.t1_mw  = 234e-3;

            objGPU = gpuMCRMWI(te, tr, fa, fixed_params);

            pars.S0      = 0.5  + 1.5  * rand(1, Nsample);
            pars.MWF     = 1e-8 + 0.25 * rand(1, Nsample);
            pars.IWF     = 0.2  + 0.6  * rand(1, Nsample);
            pars.R2sMW   = 1/20e-3 - (1/20e-3 - 1/5e-3)   * rand(1, Nsample);
            pars.R2sIW   = min(1/120e-3 - (1/120e-3 - 1/60e-3) * rand(1, Nsample), 50);
            pars.R2sEW   = min(pars.R2sIW + 20 * rand(1, Nsample), 50);
            pars.freqMW  = (0 + 15 * rand(1, Nsample)) / (objGPU.B0 * objGPU.gyro);
            pars.freqIW  = (-4 + 4 * rand(1, Nsample)) / (objGPU.B0 * objGPU.gyro);
            pars.R1IEW   = 1/2.5 + (1/0.8 - 1/2.5) * rand(1, Nsample);
            pars.kIEWM   = 1e-6 + 2 * rand(1, Nsample);
            pars.dpini   = -1 + 2 * rand(1, Nsample);
            pars.dfreqBKG = -0.05 + 0.1 * rand(1, Nsample, 1, 1, Nfa);

            fitting = [];
            fitting.DIMWI.isFitIWF    = true;
            fitting.DIMWI.isFitFreqMW = true;
            fitting.DIMWI.isFitFreqIW = true;
            fitting.DIMWI.isFitR2sEW  = true;
            fitting.isFitExchange     = true;
            fitting.isEPG             = true;
            fitting.isComplex         = true;

            spatialSize = size(pars.S0, 1:3); % [1, Nsample, 1]
            extraData.freqBKG = zeros(spatialSize);
            extraData.pini    = zeros(spatialSize);
            extraData.ff      = ones(spatialSize);
            extraData.theta   = zeros(spatialSize);
            extraData.b1      = ones(spatialSize);

            epgxDir    = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'MCRMWI', 'EPGXgen_net');
            dlnet_magn  = load(fullfile(epgxDir, 'MCRMWI_MLP_EPGX_RFphase50_T1M234_magn.mat'));
            dlnet_phase = load(fullfile(epgxDir, 'MCRMWI_MLP_EPGX_RFphase50_T1M234_phase.mat'));
            dlnet_magn.dlnet.alpha  = 0.01;
            dlnet_phase.dlnet.alpha = 0.01;

            mask = true(spatialSize);

            Sgpu_GT = gather(extractdata(objGPU.FWD(pars, fitting, extraData, dlnet_phase.dlnet, dlnet_magn.dlnet)));
            Sgpu_GT = reshape(utils.reshape_GD2ND(Sgpu_GT, mask), [1 Nsample 1 Nt Nfa 2]);
            Sgpu_GT = Sgpu_GT(:, :, :, :, :, 1) + 1i * Sgpu_GT(:, :, :, :, :, 2);

            noiseSigma = mean(pars.S0) / 100;
            y = Sgpu_GT + noiseSigma * randn(size(Sgpu_GT)) + 1i * noiseSigma * randn(size(Sgpu_GT));

            fitting              = [];
            fitting.solver       = 'askadam';
            fitting              = objGPU.check_set_default(fitting, y);
            fitting.start        = 'prior';
            fitting.iteration    = 30;

            objGPU = gpuMCRMWI(te, tr, fa, fixed_params);
            out    = objGPU.estimate(y, mask, extraData, fitting);

            testCase.verifyTrue(isfield(out, 'final'));
            for name = {'S0', 'MWF', 'IWF', 'R2sMW', 'R2sIW', 'R2sEW', 'R1IEW', 'kIEWM'}
                testCase.verifyTrue(isfield(out.final, name{1}));
                testCase.verifyTrue(all(isfinite(out.final.(name{1})(:))));
            end
        end
    end
end
