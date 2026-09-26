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
