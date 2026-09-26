classdef McmcBayesUnitTest < matlab.unittest.TestCase
    % Unit tests for the EXPERIMENTAL mcmc_bayes subclass.
    % Phase 0: legacy-option detection (mcmc_bayes.isLegacy) and the
    % not-implemented guard for Phase 2+ options.
    % Phase 1: parameter transforms, per-parameter parsing, R-hat/ESS and
    % a small GPU run of the new sampling path.
    %
    % All tests except testNewPathRuns* are pure math and need no GPU.
    %
    % Tolerances (stated before running; see each test):
    %   transform round trip        : |x - inv(fwd(x))| <= 1e-10*(ub-lb) (double), 1e-4*(ub-lb) (single)
    %   logjac vs finite difference : |logJ - log(FD)| <= 1e-6, central difference, h = 1e-5, double
    %   large |u|                   : inverse finite and inside [lb,ub] (sigmoid); logjac finite and
    %                                 |logJ - (log(ub-lb) - |u|)| <= 1e-6*|u| for |u| >= 50 (sigmoid)
    %   fused GPU kernel vs helpers : RelTol 1e-5, AbsTol 1e-6 (x) / 1e-5 (logJ), single
    %   ESS, iid (Nv=400, 4 chains x 2000)      : median ESS/(N*M) in [0.9, 1.1]
    %   ESS, AR(1) phi=0.9 (Nv=400, 4 x 4000)   : median ESS/ESS_true in [0.9, 1.1],
    %                                             ESS_true = N*M*(1-phi)/(1+phi)
    %   R-hat, iid                              : 99th percentile of |R-1| <= 0.005
    %   R-hat, AR(1) phi=0.9                    : median R <= 1.01
    %   R-hat, chains shifted by 1 SD each      : min R >= 1.1
    %
    % Kwok-Shing Chan @ MGH

    properties (TestParameter)
        % one non-default value per new option
        nonDefaultOption = struct( ...
            'parameterTransform',       {{'parameterTransform', 'sigmoid'}}, ...
            'parameterTransformCell',   {{'parameterTransform', {'linear','log'}}}, ...
            'likelihood',               {{'likelihood', 'marginal_noise'}}, ...
            'S0Param',                  {{'S0Param', 'M0'}}, ...
            'updateScheme',             {{'updateScheme', 'componentwise'}}, ...
            'adaptStepSize',            {{'adaptStepSize', true}}, ...
            'adaptInterval',            {{'adaptInterval', 100}}, ...
            'adaptTarget',              {{'adaptTarget', 0.3}}, ...
            'overdisp',                 {{'overdisp', 0.01}}, ...
            'prior',                    {{'prior', struct('hierarchical', struct())}} ...
            )

        % Phase 2+ options, still not implemented
        phase2Option = struct( ...
            'likelihood',               {{'likelihood', 'marginal_noise'}}, ...
            'S0Param',                  {{'S0Param', 'M0'}}, ...
            'prior',                    {{'prior', struct('hierarchical', struct())}} ...
            )

        transformMethod = {'linear','sigmoid','log'}
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
        function testIsLegacyEmptyStruct(testCase)
            testCase.verifyTrue(mcmc_bayes.isLegacy(struct()));
            testCase.verifyTrue(mcmc_bayes.isLegacy([]));
        end

        function testIsLegacyUnrelatedFields(testCase)
            fitting = struct('iteration', 100, 'algorithm', 'MH', 'solver', 'mcmc');
            testCase.verifyTrue(mcmc_bayes.isLegacy(fitting));
        end

        function testIsLegacyExplicitDefaults(testCase)
            fitting = McmcBayesUnitTest.explicitDefaults();
            [tf, nonDefault] = mcmc_bayes.isLegacy(fitting);
            testCase.verifyTrue(tf);
            testCase.verifyEmpty(nonDefault);

            % per-parameter cell of all 'linear' is also legacy
            fitting.parameterTransform = {'linear','Linear'};
            testCase.verifyTrue(mcmc_bayes.isLegacy(fitting));
        end

        function testIsLegacyFalseForNonDefault(testCase, nonDefaultOption)
            name = nonDefaultOption{1};

            % alone
            fitting         = struct();
            fitting.(name)  = nonDefaultOption{2};
            [tf, nonDefault] = mcmc_bayes.isLegacy(fitting);
            testCase.verifyFalse(tf);
            testCase.verifyEqual(nonDefault, {name});

            % among explicit defaults
            fitting         = McmcBayesUnitTest.explicitDefaults();
            fitting.(name)  = nonDefaultOption{2};
            [tf, nonDefault] = mcmc_bayes.isLegacy(fitting);
            testCase.verifyFalse(tf);
            testCase.verifyEqual(nonDefault, {name});
        end

        function testPhase2OptionErrorsNotImplemented(testCase, phase2Option)
            % the guard fires before any GPU code, so no GPU is needed
            fitting = struct();
            fitting.(phase2Option{1}) = phase2Option{2};
            testCase.verifyError(@() mcmc_bayes().optimisation([], [], [], [], fitting, []), ...
                'mcmc_bayes:notImplemented');

            % also together with Phase 1 options
            fitting.parameterTransform = 'sigmoid';
            fitting.updateScheme       = 'componentwise';
            testCase.verifyError(@() mcmc_bayes().optimisation([], [], [], [], fitting, []), ...
                'mcmc_bayes:notImplemented');
        end

        function testEnsembleErrorsOnNewPath(testCase)
            fitting = struct('parameterTransform', 'sigmoid', 'algorithm', 'GW');
            testCase.verifyError(@() mcmc_bayes().optimisation([], [], [], [], fitting, []), ...
                'mcmc_bayes:unsupportedAlgorithm');
        end

        function testInvalidUpdateSchemeErrors(testCase)
            fitting = struct('updateScheme', 'blockwise');
            testCase.verifyError(@() mcmc_bayes().optimisation([], [], [], [], fitting, []), ...
                'mcmc_bayes:invalidUpdateScheme');
        end

        %% Phase 1: transforms
        function testTransformRoundTrip(testCase, transformMethod)
            [lb, ub] = McmcBayesUnitTest.boundsFor(transformMethod);
            x = linspace(lb + 0.01*(ub-lb), ub - 0.01*(ub-lb), 101);

            % double
            u  = mcmc_bayes.transform_forward(x, transformMethod, lb, ub);
            x2 = mcmc_bayes.transform_inverse(u, transformMethod, lb, ub);
            testCase.verifyLessThanOrEqual(max(abs(x2 - x)), 1e-10*(ub-lb));

            % single
            xs  = single(x);
            us  = mcmc_bayes.transform_forward(xs, transformMethod, single(lb), single(ub));
            xs2 = mcmc_bayes.transform_inverse(us, transformMethod, single(lb), single(ub));
            testCase.verifyClass(xs2, 'single');
            testCase.verifyLessThanOrEqual(double(max(abs(xs2 - xs))), 1e-4*(ub-lb));
        end

        function testTransformLogJacobianFiniteDifference(testCase, transformMethod)
            [lb, ub] = McmcBayesUnitTest.boundsFor(transformMethod);
            switch transformMethod
                case 'log';     u = linspace(log(lb)+0.1, log(ub)-0.1, 41);
                case 'sigmoid'; u = linspace(-8, 8, 41);
                otherwise;      u = linspace(lb, ub, 41);
            end
            h    = 1e-5;
            fd   = (mcmc_bayes.transform_inverse(u+h, transformMethod, lb, ub) - ...
                    mcmc_bayes.transform_inverse(u-h, transformMethod, lb, ub)) ./ (2*h);
            logJ = mcmc_bayes.transform_logjac(u, transformMethod, lb, ub);
            testCase.verifyLessThanOrEqual(max(abs(logJ - log(abs(fd)))), 1e-6);
        end

        function testTransformStableAtLargeU(testCase)
            lb = 0.5; ub = 2;
            u  = [-1e4 -1e3 -100 -50 50 100 1e3 1e4];
            for cls = {'double','single'}
                uc   = cast(u, cls{1});
                x    = mcmc_bayes.transform_inverse(uc, 'sigmoid', cast(lb,cls{1}), cast(ub,cls{1}));
                logJ = mcmc_bayes.transform_logjac(uc, 'sigmoid', cast(lb,cls{1}), cast(ub,cls{1}));
                testCase.verifyTrue(all(isfinite(x)), ['sigmoid inverse not finite, ' cls{1}]);
                testCase.verifyTrue(all(x >= lb & x <= ub), ['sigmoid inverse outside box, ' cls{1}]);
                testCase.verifyTrue(all(isfinite(logJ)), ['sigmoid logjac not finite, ' cls{1}]);
                testCase.verifyLessThanOrEqual(max(abs(double(logJ) - (log(ub-lb) - abs(u))) ./ abs(u)), 1e-6);

                % log: the Jacobian is u itself, finite for any finite u
                logJ = mcmc_bayes.transform_logjac(uc, 'log', cast(0,cls{1}), cast(1,cls{1}));
                testCase.verifyEqual(logJ, uc);
            end

            % forward transform at (and beyond) the bounds is finite because of the eps clamp
            for m = {'sigmoid','log'}
                lb = 0; ub = 0.1;
                u  = mcmc_bayes.transform_forward([lb-1 lb ub ub+1], m{1}, lb, ub);
                testCase.verifyTrue(all(isfinite(u)), ['forward not finite at bounds, ' m{1}]);
                u  = mcmc_bayes.transform_forward(single([lb ub]), m{1}, single(lb), single(ub));
                testCase.verifyTrue(all(isfinite(u)), ['forward not finite at bounds (single), ' m{1}]);
            end
        end

        function testTransformPerParameterCell(testCase)
            % string applies to all parameters
            testCase.verifyEqual(mcmc_bayes.parse_transform('Sigmoid', 3), {'sigmoid','sigmoid','sigmoid'});
            testCase.verifyEqual(mcmc_bayes.parse_transform("log", 2), {'log','log'});
            % cell: one entry per parameter, case-insensitive
            testCase.verifyEqual(mcmc_bayes.parse_transform({'linear','Log','SIGMOID'}, 3), {'linear','log','sigmoid'});
            % wrong length or unknown name
            testCase.verifyError(@() mcmc_bayes.parse_transform({'linear','log'}, 3), 'mcmc_bayes:invalidTransform');
            testCase.verifyError(@() mcmc_bayes.parse_transform('logit', 3), 'mcmc_bayes:invalidTransform');
            testCase.verifyError(@() mcmc_bayes.parse_transform(1, 3), 'mcmc_bayes:invalidTransform');

            % bounds
            testCase.verifyError(@() mcmc_bayes.check_transform_bounds({'log'}, -1, 1), 'mcmc_bayes:invalidBounds');
            testCase.verifyError(@() mcmc_bayes.check_transform_bounds({'sigmoid'}, 0, Inf), 'mcmc_bayes:invalidBounds');
            mcmc_bayes.check_transform_bounds({'linear','log','sigmoid'}, [-Inf 0 0], [Inf 1 1]);    % no error

            % rows are transformed independently with their own method and bounds
            method  = {'linear','sigmoid','log'};
            lb      = [0; 0.5; 0.001];
            ub      = [1; 2;   200];
            x       = [0.3 0.7; 1.2 0.9; 30 0.05];
            u       = mcmc_bayes.transform_forward(x, method, lb, ub);
            for k = 1:3
                testCase.verifyEqual(u(k,:), mcmc_bayes.transform_forward(x(k,:), method{k}, lb(k), ub(k)));
            end
            logJ    = mcmc_bayes.transform_logjac(u, method, lb, ub);
            testCase.verifyEqual(logJ(1,:), [0 0]);
            testCase.verifyEqual(logJ(3,:), u(3,:));
            testCase.verifyEqual(mcmc_bayes.transform_inverse(u, method, lb, ub), x, 'AbsTol', 1e-10);
        end

        function testFusedKernelMatchesHelpers(testCase)
            % the sampler uses a fused GPU kernel; it must match the reference helpers
            gacelletest.assumeGPU(testCase);
            method  = {'linear','sigmoid','log'};
            lb      = gpuArray(single([0; 0.5; 0.001]));
            ub      = gpuArray(single([2; 2;   200]));
            u       = gpuArray(single([linspace(0.1,1.9,201); linspace(-60,60,201); linspace(-6,5,201)]));
            code    = gpuArray(single(mcmc_bayes.transform_code(method)));
            [x, logJ] = mcmc_bayes.transform_inverse_logjac_fused(u, code, lb, ub);
            xRef    = mcmc_bayes.transform_inverse(u, method, lb, ub);
            logJRef = mcmc_bayes.transform_logjac(u, method, lb, ub);
            testCase.verifyClass(gather(x), 'single');
            testCase.verifyEqual(gather(x),    gather(xRef),    'RelTol', single(1e-5), 'AbsTol', single(1e-6));
            testCase.verifyEqual(gather(logJ), gather(logJRef), 'RelTol', single(1e-5), 'AbsTol', single(1e-5));
        end

        %% Phase 1: diagnostics
        function testEssRhatIid(testCase)
            rng(20260926);
            Nv = 400; N = 2000; M = 4;
            x  = randn(Nv, N, M);
            neff = mcmc_bayes.ess(x);
            R    = mcmc_bayes.rhat(x);
            testCase.verifySize(neff, [Nv 1]);
            testCase.verifySize(R,    [Nv 1]);
            ratio = median(neff) / (N*M);
            testCase.verifyGreaterThanOrEqual(ratio, 0.9, sprintf('iid ESS ratio %.3f', ratio));
            testCase.verifyLessThanOrEqual(ratio, 1.1, sprintf('iid ESS ratio %.3f', ratio));
            dR = sort(abs(R-1));    % 99th percentile without the Statistics toolbox
            testCase.verifyLessThanOrEqual(dR(ceil(0.99*Nv)), 0.005);
        end

        function testEssRhatAR1(testCase)
            rng(20260927);
            Nv = 400; N = 4000; M = 4; phi = 0.9; Nskip = 500;
            e  = randn(Nv, N+Nskip, M);
            x  = filter(sqrt(1-phi^2), [1 -phi], e, [], 2);   % stationary variance 1
            x  = x(:, Nskip+1:end, :);
            essTrue = N*M*(1-phi)/(1+phi);
            neff  = mcmc_bayes.ess(x);
            ratio = median(neff) / essTrue;
            testCase.verifyGreaterThanOrEqual(ratio, 0.9, sprintf('AR(1) ESS ratio %.3f', ratio));
            testCase.verifyLessThanOrEqual(ratio, 1.1, sprintf('AR(1) ESS ratio %.3f', ratio));
            testCase.verifyLessThanOrEqual(median(mcmc_bayes.rhat(x)), 1.01);
        end

        function testRhatDetectsNonMixing(testCase)
            rng(20260928);
            Nv = 100; N = 1000; M = 4;
            x  = randn(Nv, N, M) + reshape(0:M-1, 1, 1, M);  % chain m shifted by (m-1) SD
            testCase.verifyGreaterThanOrEqual(min(mcmc_bayes.rhat(x)), 1.1);
        end

        %% Phase 1: new sampling path on the GPU
        function testNewPathRunsAllPhase1Options(testCase)
            gacelletest.assumeGPU(testCase);
            [y, mask, w, pars0, fitting, obj] = McmcBayesUnitTest.r2starSetup();

            rng(1); parallel.gpu.rng(1);
            outRef = mcmc().optimisation(y, mask, w, pars0, fitting, @obj.FWD, 'mcmc', fitting);

            f = fitting;
            f.parameterTransform = {'sigmoid','log','log'};
            f.updateScheme       = 'componentwise';
            f.adaptStepSize      = true;
            f.adaptInterval      = 20;
            f.overdisp           = 0.01;
            f.repetition         = 2;
            out = mcmc_bayes().optimisation(y, mask, w, pars0, f, @obj.FWD, 'mcmc', f);

            params = fitting.modelParams;
            for k = 1:numel(params)
                p  = params{k};
                sz = size(outRef.posterior.(p)); sz(3) = 2;
                testCase.verifyEqual(size(out.posterior.(p)), sz);
                testCase.verifyGreaterThanOrEqual(min(out.posterior.(p)(:)), single(fitting.lb(k)));
                testCase.verifyLessThanOrEqual(max(out.posterior.(p)(:)), single(fitting.ub(k)));
                testCase.verifyTrue(all(isfinite(out.mean.(p)(:))));
                testCase.verifyTrue(isfield(out.diagnostics.rhat, p));
                testCase.verifyTrue(isfield(out.diagnostics.ess,  p));
                testCase.verifyTrue(isfield(out.diagnostics.stepSize, p));
            end
            testCase.verifyEqual(size(out.diagnostics.acceptance), [size(mask,1:3) numel(params) 2]);
            testCase.verifyEqual(out.settings.parameterTransform, {'sigmoid','log','log'});
            testCase.verifyEqual(out.settings.adaptTarget, 0.44);
        end

        function testNewPathShortBurninWarns(testCase)
            gacelletest.assumeGPU(testCase);
            [y, mask, w, pars0, fitting, obj] = McmcBayesUnitTest.r2starSetup();
            f = fitting;
            f.adaptStepSize = true;
            f.adaptInterval = 200;      % Nburnin = 20 < 2*200
            testCase.verifyWarning(@() mcmc_bayes().optimisation(y, mask, w, pars0, f, @obj.FWD, 'mcmc', f), ...
                'mcmc_bayes:shortBurnin');
        end
    end

    methods (Static)
        function [lb, ub] = boundsFor(method)
            switch method
                case 'log';     lb = 0.001; ub = 200;
                otherwise;      lb = 0.5;   ub = 2;
            end
        end

        % tiny R2* dataset as in SmokeFit_R2starMappingTest / McmcBayesLegacyTest
        function [y, mask, w, pars0, fitting, obj] = r2starSetup()
            seed = 1; rng(seed); gpurng(seed);
            Nsample = 8;
            te      = linspace(0, 40e-3, 6);
            pars.M0     = 1   + 0.1 * rand(1, Nsample);
            pars.R2star = 30  + 5   * rand(1, Nsample);
            obj     = gpuR2starMapping(te);
            s       = obj.FWD(pars);
            y       = s + mean(pars.M0)/50 * randn(size(s));
            y       = permute(y, [2 3 4 1]);
            mask    = true(size(y, 1:3));

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
            fitting.ub          = obj.ub(1:numel(obj.modelParams));
            fitting.lb          = obj.lb(1:numel(obj.modelParams));
            pars0               = obj.determine_x0(y, mask, fitting);
            w                   = obj.compute_optimisation_weights(y, fitting);
            fitting.xStepSize   = obj.step;
        end

        function fitting = explicitDefaults()
            fitting.parameterTransform  = 'linear';
            fitting.likelihood          = 'gaussian';
            fitting.S0Param             = '';
            fitting.updateScheme        = 'joint';
            fitting.adaptStepSize       = false;
            fitting.adaptInterval       = 50;
            fitting.adaptTarget         = [];
            fitting.overdisp            = 0;
            fitting.prior               = [];
        end
    end
end
