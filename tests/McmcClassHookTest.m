classdef McmcClassHookTest < matlab.unittest.TestCase
    % Tier-2 (GPU) tests of the fitting.mcmcClass hook in the model wrappers not covered by
    % McmcBayesLegacyTest (NEXI, JointR1R2starMapping, GREMWI, mcmicro, IVIM; gpuMCRMWI does not
    % support mcmc):
    %   - with mcmcClass = 'mcmc_bayes' and no new option, estimate() is bitwise identical to
    %     the default hook ('mcmc');
    %   - with a mcmc_bayes-only option (likelihood = 'marginal_noise') the hook reaches
    %     mcmc_bayes (out.settings.likelihood is set) and the fit is finite;
    %   - a prior that couples voxels with NSegmentUser = 2 errors <Class>:singleSegment;
    %   - FWD with the legacy ensemble label 'GW' reshapes per walker as with 'ensemble'.
    % Data: the tiny synthetic datasets of the SmokeFit_*Test of each model.
    %
    % Kwok-Shing Chan @ MGH

    properties (TestParameter)
        model = {'NEXI','JointR1R2starMapping','GREMWI','mcmicro','IVIM'}
        ensembleModel = {'JointR1R2starMapping','GREMWI'}
    end

    methods (TestClassSetup)
        function addGacellePath(testCase)
            testsDir    = fileparts(mfilename('fullpath'));
            projectRoot = fileparts(testsDir);
            addpath(projectRoot);
            addpath_gacelle(projectRoot);
        end
    end

    methods (Test)
        function testHookBitwise(testCase, model)
            gacelletest.assumeGPU(testCase);
            [y, mask, extraData, obj, f] = McmcClassHookTest.setup(model);
            runSeed = 48463;
            rng(runSeed); parallel.gpu.rng(runSeed);
            outRef = obj().estimate(y, mask, extraData, f);
            f.mcmcClass = 'mcmc_bayes';
            rng(runSeed); parallel.gpu.rng(runSeed);
            outNew = obj().estimate(y, mask, extraData, f);
            testCase.verifyTrue(isequaln(outNew, outRef), 'wrapper output differs between mcmcClass ''mcmc'' and ''mcmc_bayes''');
        end

        function testHookReachesMcmcBayes(testCase, model)
            gacelletest.assumeGPU(testCase);
            [y, mask, extraData, obj, f] = McmcClassHookTest.setup(model);
            f.mcmcClass = 'mcmc_bayes'; f.likelihood = 'marginal_noise';
            out = obj().estimate(y, mask, extraData, f);
            testCase.verifyEqual(out.settings.likelihood, 'marginal_noise');
            fn = setdiff(fieldnames(out.mean), {'noise'});
            for k = 1:numel(fn)
                testCase.verifyTrue(all(isfinite(out.mean.(fn{k})(:))), fn{k});
            end
        end

        function testSingleSegmentGuard(testCase, model)
            gacelletest.assumeGPU(testCase);
            [y, mask, extraData, obj, f] = McmcClassHookTest.setup(model);
            % 4 slices so that NSegmentUser = 2 really splits the volume
            y = repmat(y, [1 1 4 ones(1, ndims(y)-3)]); mask = repmat(mask, [1 1 4]);
            if isstruct(extraData)
                for fn = fieldnames(extraData)'; extraData.(fn{1}) = repmat(extraData.(fn{1}), [1 1 4]); end
            end
            f.iteration = 20; f.mcmcClass = 'mcmc_bayes'; f.NSegmentUser = 2;
            f.likelihood = 'marginal_noise'; f.parameterTransform = 'sigmoid';
            f.prior = struct('hierarchical', struct('params', {McmcClassHookTest.hierParam(model)}));
            testCase.verifyError(@() obj().estimate(y, mask, extraData, f), ['gpu' model ':singleSegment']);
        end

        function testEnsembleLegacyLabelFWD(testCase, ensembleModel)
            % FWD called directly with the legacy label 'GW' reshapes per walker exactly as with
            % 'ensemble' (estimate() converts the label itself, so this is the direct-call path)
            gacelletest.assumeGPU(testCase);
            Nv = 4; Nw = 6; N = Nv*Nw; rng(2);
            switch ensembleModel
                case 'JointR1R2starMapping'
                    obj = gpuJointR1R2starMapping(linspace(1.5e-3, 42e-3, 5), 45e-3, [5 20 50]);
                    p   = struct('M0', gpuArray(single(1 + rand(1,N))), 'R1', gpuArray(single(0.5 + rand(1,N))), ...
                                 'R2star', gpuArray(single(20 + 20*rand(1,N))));
                    x   = struct('b1', gpuArray(ones(1,N,'single')));
                    fwd = @(alg) obj.FWD(p, x, 'mcmc', struct('algorithm', alg, 'Nwalker', Nw));
                case 'GREMWI'
                    obj = gpuGREMWI(linspace(1.5e-3, 42e-3, 6));
                    B0  = 3;
                    p   = struct('S0', 1 + rand(1,N), 'MWF', 0.1*rand(1,N), 'IWF', 0.2 + 0.6*rand(1,N), ...
                                 'R2sMW', 75 + 75*rand(1,N), 'R2sIW', 10 + 20*rand(1,N), 'R2sEW', 20 + 20*rand(1,N), ...
                                 'freqMW', 10*rand(1,N)/B0/gpuGREMWI.gyro, 'freqIW', -5*rand(1,N)/B0/gpuGREMWI.gyro, ...
                                 'dfreqBKG', 0.01*rand(1,N), 'dpini', 0.1*rand(1,N));
                    x   = struct('freqBKG', zeros(1,N), 'pini', zeros(1,N), 'ff', ones(1,N), 'theta', zeros(1,N));
                    p   = structfun(@(v) gpuArray(single(v)), p, 'UniformOutput', false);    % as passed by mcmc
                    x   = structfun(@(v) gpuArray(single(v)), x, 'UniformOutput', false);
                    fs  = struct('isComplex', true, 'solver', 'mcmc', 'Nwalker', Nw, 'DIMWI', ...
                                 struct('isFitIWF', true, 'isFitFreqMW', true, 'isFitFreqIW', true, 'isFitR2sEW', true));
                    fwd = @(alg) obj.FWD(p, setfield(fs, 'algorithm', alg), x); %#ok<SFLD>
            end
            sE = fwd('ensemble');
            sG = fwd('GW');
            testCase.verifyEqual(size(sE, 2:3), [Nv Nw]);
            testCase.verifyEqual(gather(sG), gather(sE));
        end
    end

    methods (Static)
        function p = hierParam(model)
            % one fitted parameter per model for the hierarchical prior
            switch model
                case 'NEXI';                    p = {'fa'};
                case 'JointR1R2starMapping';    p = {'R2star'};
                case 'GREMWI';                  p = {'MWF'};
                case {'mcmicro','IVIM'};        p = {'f'};
            end
        end

        function [y, mask, extraData, obj, f] = setup(model)
            seed = 1; rng(seed); gpurng(seed);
            f = struct('solver','mcmc', 'algorithm','MH', 'iteration',100, 'thinning',2, 'burnin',0.1, ...
                       'metric',{{'mean','std'}});
            extraData = [];
            switch model
                case 'NEXI'
                    Nsample = 8; bval = [2.3, 3.5, 4.8, 6.5, 2.3, 3.5]; BDELTA = [13, 13, 13, 13, 21, 21];
                    pars.fa = 0.1 + 0.7 * rand(1, Nsample);
                    pars.Da = 1.5 + 1.5 * rand(1, Nsample);
                    pars.De = 0.5 + 1.0 * rand(1, Nsample);
                    tex     = 2   + 48  * rand(1, Nsample);
                    pars.ra = (1 - pars.fa) ./ tex;
                    pars.p2 = 0.05 + 0.45 * rand(1, Nsample);
                    obj = @() gpuNEXI(bval, BDELTA);
                    o = obj(); s = o.FWD(pars, 2);
                    y = permute(s + randn(size(s))/50, [3 2 4 1]);
                    f.lmax = 2; f.start = 'likelihood';

                case 'JointR1R2starMapping'
                    Nsample = 6; Nt = 5; t = linspace(1.5e-3, 42e-3, Nt); tr = 45e-3; fa = [5, 20, 50];
                    pars.M0     = 0.5 + 1.5 * rand(1, Nsample);
                    pars.R1     = 0.2 + 2.3 * rand(1, Nsample);
                    pars.R2star = 5   + 55  * rand(1, Nsample);
                    b1 = ones(size(pars.M0));
                    obj = @() gpuJointR1R2starMapping(t, tr, fa);
                    o = obj(); s = gather(o.FWD(pars, struct('b1', b1)));
                    s = permute(reshape(s, [Nt, numel(fa), Nsample]), [3 4 5 1 2]);
                    y = s + (pars.M0.')/100 .* randn(size(s));
                    extraData.b1 = b1.';
                    f.start = 'default';

                case 'GREMWI'
                    Nsample = 6; Nt = 6; t = linspace(1.5e-3, 42e-3, Nt); B0 = 3;
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
                    fs = struct('isComplex', true, 'DIMWI', struct('isFitIWF', true, 'isFitFreqMW', true, ...
                                'isFitFreqIW', true, 'isFitR2sEW', true));
                    sp = size(pars.S0, 1:3);
                    extraData = struct('freqBKG', zeros(sp), 'pini', zeros(sp), 'ff', ones(sp), 'theta', zeros(sp));
                    obj = @() gpuGREMWI(t);
                    o = obj(); s = (o.FWD(pars, fs, extraData)).';
                    s = permute(reshape(s, [Nsample, Nt, 2]), [1 4 5 2 3]);
                    y = s + (pars.S0.')/150 .* randn(size(s));
                    y = y(:,:,:,:,1) + 1i * y(:,:,:,:,2);
                    f.start = 'prior';

                case 'IVIM'
                    [y, ~, bval] = SmokeFit_IVIMTest.data();
                    obj = @() gpuIVIM(bval);

                case 'mcmicro'
                    Nsample = 8; bval = [0, 0.5, 1, 2];
                    pars.f = 0.1 + 0.7 * rand(1, Nsample);
                    pars.D = 1.5 + 1.5 * rand(1, Nsample);
                    obj = @() gpumcmicro(bval);
                    o = obj(); s = o.FWD(pars);
                    y = permute(s + randn(size(s))/40, [3 2 4 1]);
            end
            y    = gather(y);
            mask = true(size(y, 1:3));
        end
    end
end
