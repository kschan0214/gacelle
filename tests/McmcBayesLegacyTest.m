classdef McmcBayesLegacyTest < matlab.unittest.TestCase
    % Tier-2 (GPU) legacy-identity test for the EXPERIMENTAL mcmc_bayes
    % subclass: with all new options absent or at their defaults,
    % mcmc_bayes().optimisation must return output bitwise identical to
    % mcmc().optimisation for the same seeds.
    %
    % Uses the same tiny synthetic R2* dataset as
    % SmokeFit_R2starMappingTest and replicates the preparation done in
    % gpuR2starMapping.fit before it calls mcmc().optimisation.
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
        function testDefaultsBitwiseIdenticalToMcmc(testCase)
            testCase.checkBitwiseIdentical(struct());
        end

        function testR2starWrapperHookBitwise(testCase)
            % Phase 6a: gpuR2starMapping.estimate with fitting.mcmcClass = 'mcmc_bayes' (and no new
            % option) must be bitwise identical to the default hook ('mcmc')
            gacelletest.assumeGPU(testCase);
            [y, mask, te] = McmcBayesLegacyTest.r2starData();
            f = struct('solver','mcmc', 'algorithm','MH', 'iteration',200, 'thinning',2, 'burnin',0.1, ...
                       'metric',{{'mean','std'}}, 'start','default');
            runSeed = 48463;
            rng(runSeed); parallel.gpu.rng(runSeed);
            outRef = gpuR2starMapping(te).estimate(y, mask, f);
            f.mcmcClass = 'mcmc_bayes';
            rng(runSeed); parallel.gpu.rng(runSeed);
            outNew = gpuR2starMapping(te).estimate(y, mask, f);
            testCase.verifyTrue(isequaln(outNew, outRef), 'wrapper output differs between mcmcClass ''mcmc'' and ''mcmc_bayes''');
        end

        function testAxCaliberSMTWrapperHookBitwise(testCase)
            % Phase 6c: gpuAxCaliberSMT.estimate with fitting.mcmcClass = 'mcmc_bayes' (and no new
            % option) must be bitwise identical to the default hook ('mcmc')
            gacelletest.assumeGPU(testCase);
            [y, mask, obj] = McmcBayesLegacyTest.axcaliberData();
            f = struct('solver','mcmc', 'algorithm','MH', 'iteration',200, 'thinning',2, 'burnin',0.1, ...
                       'metric',{{'mean','std'}}, 'start','default');
            runSeed = 48463;
            rng(runSeed); parallel.gpu.rng(runSeed);
            outRef = obj().estimate(y, mask, [], f);
            f.mcmcClass = 'mcmc_bayes';
            rng(runSeed); parallel.gpu.rng(runSeed);
            outNew = obj().estimate(y, mask, [], f);
            testCase.verifyTrue(isequaln(outNew, outRef), 'wrapper output differs between mcmcClass ''mcmc'' and ''mcmc_bayes''');
        end

        function testAxCaliberSMTWrapperSingleSegmentGuard(testCase)
            gacelletest.assumeGPU(testCase);
            [y, mask, obj] = McmcBayesLegacyTest.axcaliberData();
            y = repmat(y, [1 1 4 1]); mask = repmat(mask, [1 1 4]);
            f = struct('solver','mcmc', 'algorithm','MH', 'iteration',20, 'mcmcClass','mcmc_bayes', 'NSegmentUser', 2, ...
                       'likelihood','marginal_noise', 'parameterTransform','sigmoid', 'start','default', ...
                       'prior', struct('hierarchical', struct('params', {{'a','f','fcsf','DeR'}})));
            testCase.verifyError(@() obj().estimate(y, mask, [], f), 'gpuAxCaliberSMT:singleSegment');
        end

        function testSANDIWrapperHookBitwise(testCase)
            % Phase 6d: gpuSANDI.estimate with fitting.mcmcClass = 'mcmc_bayes' (and no new option)
            % must be bitwise identical to the default hook ('mcmc')
            gacelletest.assumeGPU(testCase);
            [y, mask, obj] = McmcBayesLegacyTest.sandiData();
            f = struct('solver','mcmc', 'algorithm','MH', 'iteration',200, 'thinning',2, 'burnin',0.1, ...
                       'metric',{{'mean','std'}}, 'start','default');
            runSeed = 48463;
            rng(runSeed); parallel.gpu.rng(runSeed);
            outRef = obj().estimate(y, mask, [], f);
            f.mcmcClass = 'mcmc_bayes';
            rng(runSeed); parallel.gpu.rng(runSeed);
            outNew = obj().estimate(y, mask, [], f);
            testCase.verifyTrue(isequaln(outNew, outRef), 'wrapper output differs between mcmcClass ''mcmc'' and ''mcmc_bayes''');
        end

        function testSANDIWrapperSingleSegmentGuard(testCase)
            gacelletest.assumeGPU(testCase);
            [y, mask, obj] = McmcBayesLegacyTest.sandiData();
            y = repmat(y, [1 1 4 1]); mask = repmat(mask, [1 1 4]);
            f = struct('solver','mcmc', 'algorithm','MH', 'iteration',20, 'mcmcClass','mcmc_bayes', 'NSegmentUser', 2, ...
                       'likelihood','marginal_noise', 'parameterTransform','sigmoid', 'start','default', ...
                       'prior', struct('hierarchical', struct('params', {{'Rs','fs','f','Da','De'}})));
            testCase.verifyError(@() obj().estimate(y, mask, [], f), 'gpuSANDI:singleSegment');
        end

        function testMEAxCaliberSMTWrapperHookBitwise(testCase)
            % gpuMEAxCaliberSMT with fitting.mcmcClass = 'mcmc_bayes' (and no new option)
            % must be bitwise identical to the default hook ('mcmc')
            gacelletest.assumeGPU(testCase);
            [y, mask, obj] = McmcBayesLegacyTest.meaxcaliberData();
            f = struct('solver','mcmc', 'algorithm','MH', 'iteration',200, 'thinning',2, 'burnin',0.1, ...
                       'metric',{{'mean','std'}}, 'start','default');
            runSeed = 48463;
            rng(runSeed); parallel.gpu.rng(runSeed);
            outRef = obj().estimate(y, mask, f, []);
            f.mcmcClass = 'mcmc_bayes';
            rng(runSeed); parallel.gpu.rng(runSeed);
            outNew = obj().estimate(y, mask, f, []);
            testCase.verifyTrue(isequaln(outNew, outRef), 'wrapper output differs between mcmcClass ''mcmc'' and ''mcmc_bayes''');
        end

        function testMEAxCaliberSMTWrapperSingleSegmentGuard(testCase)
            gacelletest.assumeGPU(testCase);
            [y, mask, obj] = McmcBayesLegacyTest.meaxcaliberData();
            y = repmat(y, [1 1 4 1]); mask = repmat(mask, [1 1 4]);
            f = struct('solver','mcmc', 'algorithm','MH', 'iteration',20, 'mcmcClass','mcmc_bayes', 'NSegmentUser', 2, ...
                       'likelihood','marginal_noise', 'parameterTransform','sigmoid', 'start','default', ...
                       'prior', struct('hierarchical', struct('params', {{'f','fcsf','DeR','r','R2e'}})));
            testCase.verifyError(@() obj().estimate(y, mask, f, []), 'gpuMEAxCaliberSMT:singleSegment');
        end

        function testR2starWrapperSingleSegmentGuard(testCase)
            % a voxel-coupling prior with the data divided into segments must error
            gacelletest.assumeGPU(testCase);
            [y, mask, te] = McmcBayesLegacyTest.r2starData();
            y = repmat(y, [1 1 4 1]); mask = repmat(mask, [1 1 4]);
            f = struct('solver','mcmc', 'algorithm','MH', 'iteration',20, 'mcmcClass','mcmc_bayes', 'NSegmentUser', 2, ...
                       'likelihood','marginal_S0noise', 'S0Param','M0', 'parameterTransform','sigmoid', ...
                       'prior', struct('hierarchical', struct('params', {{'R2star'}})));
            testCase.verifyError(@() gpuR2starMapping(te).estimate(y, mask, f), 'gpuR2starMapping:singleSegment');
        end

        function testExplicitDefaultsBitwiseIdenticalToMcmc(testCase)
            opts.parameterTransform = 'linear';
            opts.likelihood         = 'gaussian';
            opts.S0Param            = '';
            opts.updateScheme       = 'joint';
            opts.adaptStepSize      = false;
            opts.adaptInterval      = 50;
            opts.adaptTarget        = [];
            opts.overdisp           = 0;
            opts.prior              = [];
            testCase.checkBitwiseIdentical(opts);
        end
    end

    methods (Static)
        function [y, mask, obj] = meaxcaliberData()
            % small synthetic two-TE spherical-mean data for gpuMEAxCaliberSMT
            rng(1); gpurng(1);
            b0 = [0 0.35 1.5 3.45 6]; b = [b0 b0]; d = 6*ones(size(b)); D = 13*ones(size(b)); te = [0.051*ones(size(b0)) 0.092*ones(size(b0))];
            obj  = @() gpuMEAxCaliberSMT(b, d, D, te, [], []);
            pars = struct('f', 0.3 + 0.5*rand(1,8), 'fcsf', 0.1*rand(1,8), 'DeR', 0.3 + 0.8*rand(1,8), ...
                          'r', 0.5 + 2*rand(1,8), 'R2e', 15 + 20*rand(1,8));
            s    = obj().FWD(pars);
            s    = s ./ s(1,:);
            y    = double(permute(s + randn(size(s))/100, [3 2 4 1]));
            mask = true(size(y, 1:3));
        end

        function [y, mask, obj] = sandiData()
            % small synthetic spherical-mean data, as in SmokeFit_SANDITest; obj is a constructor handle
            rng(1); gpurng(1);
            b = [0.05, 0.3, 0.8, 1.5, 2.3, 3.5]; D = 13*ones(size(b)); d = 6*ones(size(b));
            obj  = @() gpuSANDI(b, d, D, 3);
            pars = struct('f', 0.1 + 0.7*rand(1,8), 'Da', 1.5 + 1.5*rand(1,8), 'De', 0.5 + 1.0*rand(1,8), ...
                          'Rs', 5 + 5*rand(1,8), 'fs', 0.1 + 0.7*rand(1,8));
            s    = obj().FWD(pars, 'wide');
            y    = double(permute(s + randn(size(s))/50, [3 2 4 1]));
            mask = true(size(y, 1:3));
        end

        function [y, mask, obj] = axcaliberData()
            % small synthetic spherical-mean data, as in SmokeFit_AxCaliberSMTTest; obj is a constructor handle
            rng(1); gpurng(1);
            b = [0.05, 0.35, 0.80, 1.5, 2.401, 3.45]; d = 6*ones(size(b)); D = 13*ones(size(b));
            obj  = @() gpuAxCaliberSMT(b, d, D, 1.7, 1.7, 1.7, 3);
            pars = struct('a', 0.5 + 5.5*rand(1,8), 'f', 0.3 + 0.7*rand(1,8), 'fcsf', 0.3*rand(1,8), 'DeR', 0.5 + 1.0*rand(1,8));
            s    = obj().FWD(pars, 'VanGelderen');
            y    = double(permute(s + randn(size(s))/100, [2 3 4 1]));
            mask = true(size(y, 1:3));
        end

        function [y, mask, te] = r2starData()
            % small synthetic multi-echo data, as in SmokeFit_R2starMappingTest
            rng(1); gpurng(1);
            te   = linspace(0, 40e-3, 6);
            pars = struct('M0', 1 + 0.1*rand(1,8), 'R2star', 30 + 5*rand(1,8));
            s    = gpuR2starMapping(te).FWD(pars);
            y    = s + mean(pars.M0)/50 * randn(size(s));
            y    = double(permute(y, [2 3 4 1]));   % [x,y,z,TE]
            mask = true(size(y, 1:3));
        end
    end

    methods
        function checkBitwiseIdentical(testCase, extraOpts)
            gacelletest.assumeGPU(testCase);

            % synthetic data, as in SmokeFit_R2starMappingTest
            seed = 1; rng(seed); gpurng(seed);

            Nsample = 8;
            te      = linspace(0, 40e-3, 6); % s

            pars.M0     = 1   + 0.1 * rand(1, Nsample);
            pars.R2star = 30  + 5   * rand(1, Nsample);

            obj     = gpuR2starMapping(te);
            s       = obj.FWD(pars);
            noise   = mean(pars.M0) / 50;
            y       = s + noise * randn(size(s));
            y       = permute(y, [2 3 4 1]); % [x,y,z,TE]
            mask    = true(size(y, 1:3));

            % fitting setup, replicating gpuR2starMapping.fit (mcmc branch)
            fitting             = [];
            fitting.solver      = 'mcmc';
            fitting.algorithm   = 'MH';
            fitting.iteration   = 200;
            fitting.thinning    = 2;
            fitting.burnin      = 0.1;
            fitting.metric      = {'mean','std','median'};
            fitting.start       = 'default';
            fitting             = obj.check_set_default(fitting);
            obj                 = obj.updateProperty(fitting);
            fitting.modelParams = obj.modelParams;
            if isempty(fitting.ub); fitting.ub = obj.ub(1:numel(obj.modelParams)); end
            if isempty(fitting.lb); fitting.lb = obj.lb(1:numel(obj.modelParams)); end
            pars0               = obj.determine_x0(y, mask, fitting);
            w                   = obj.compute_optimisation_weights(y, fitting);
            fitting.xStepSize   = obj.step;

            % new options (absent or defaults)
            fn = fieldnames(extraOpts);
            for k = 1:numel(fn); fitting.(fn{k}) = extraOpts.(fn{k}); end

            runSeed = 48463;

            rng(runSeed); parallel.gpu.rng(runSeed);
            outRef  = mcmc().optimisation(y, mask, w, pars0, fitting, @obj.FWD, fitting.solver, fitting);

            rng(runSeed); parallel.gpu.rng(runSeed);
            outNew  = mcmc_bayes().optimisation(y, mask, w, pars0, fitting, @obj.FWD, fitting.solver, fitting);

            % same set of output fields
            testCase.verifyEqual(sort(fieldnames(outNew)), sort(fieldnames(outRef)));

            % posterior samples
            params = fieldnames(outRef.posterior);
            testCase.verifyEqual(sort(fieldnames(outNew.posterior)), sort(params));
            for k = 1:numel(params)
                testCase.verifyTrue(isequal(outNew.posterior.(params{k}), outRef.posterior.(params{k})), ...
                    sprintf('posterior.%s differs between mcmc and mcmc_bayes', params{k}));
            end

            % metric maps
            for km = 1:numel(fitting.metric)
                metric = fitting.metric{km};
                testCase.assertTrue(isfield(outRef, metric) && isfield(outNew, metric), ...
                    sprintf('metric %s missing from output', metric));
                for k = 1:numel(params)
                    testCase.verifyTrue(isequal(outNew.(metric).(params{k}), outRef.(metric).(params{k})), ...
                        sprintf('%s.%s differs between mcmc and mcmc_bayes', metric, params{k}));
                end
            end

            % the chain must actually have moved, otherwise identity is trivial
            testCase.verifyGreaterThan(numel(unique(outRef.posterior.R2star(:))), 1);
        end
    end
end
