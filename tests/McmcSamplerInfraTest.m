classdef McmcSamplerInfraTest < matlab.unittest.TestCase
    % Tests of the opt-in sampler options of mcmc ('MH' only, Phase 8 of the mcmc_bayes plan):
    % parameterTransform, updateScheme, adaptStepSize/adaptInterval/adaptTarget, adaptCovariance,
    % overdisp, R-hat/ESS diagnostics and the forward-size check, and of their static helpers,
    % which moved here from mcmc_bayes (mcmc_bayes inherits them).
    %
    % Legacy identity (GPU): with no option set, mcmc().optimisation is bitwise identical to the
    %   legacy loop (mcmc.metropolis_hastings + res2out, called directly with the same seeds) and
    %   has no new output field; also with every option given explicitly at its default.
    %   (The identity against the pre-Phase-8 code, commit 84c1807, was checked once by a one-off
    %   script; see the Phase 8 report.)
    % mcmc vs mcmc_bayes (GPU): mcmc with the options and mcmc_bayes with the same options (Gaussian
    %   likelihood, no prior) run the same chain for the same seed: posterior, metrics and
    %   out.diagnostics bitwise identical, and the shared out.settings fields equal.
    % Option detection and errors (no GPU): mcmc.use_sampler_infra, mcmc:unsupportedAlgorithm,
    %   mcmc:invalidUpdateScheme, mcmc:adaptCovariance, mcmc:noNoise, mcmc:forwardSize.
    %
    % Tolerances of the helper tests (stated before running, unchanged from McmcBayesUnitTest):
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
    %   batched Cholesky vs chol                : <= 1e-10 * max|L| (double, CPU and GPU)
    %   Welford vs mean/cov                     : RelTol 1e-10 (mean), AbsTol 1e-10*max(diag(C)) (cov)
    %
    % Kwok-Shing Chan @ MGH

    properties (TestParameter)
        transformMethod = {'linear','sigmoid','log'}

        % opt-in option sets compared between mcmc and mcmc_bayes
        infraConfig = struct( ...
            'sigmoidAdaptCovOverdisp',  {struct('parameterTransform', 'sigmoid', 'adaptStepSize', true, ...
                                                'adaptCovariance', true, 'overdisp', 0.1, 'repetition', 2)}, ...
            'mixedComponentwise',       {struct('parameterTransform', {{'sigmoid','log','linear'}}, ...
                                                'updateScheme', 'componentwise', 'adaptStepSize', true, 'repetition', 2)}, ...
            'sigmoidOnly',              {struct('parameterTransform', 'sigmoid')}, ...
            'linearOverdisp',           {struct('overdisp', 0.05, 'repetition', 2)} ...
            )

        % one non-default value per opt-in option
        nonDefaultOption = struct( ...
            'parameterTransform',       {{'parameterTransform', 'sigmoid'}}, ...
            'parameterTransformCell',   {{'parameterTransform', {'linear','log'}}}, ...
            'updateScheme',             {{'updateScheme', 'componentwise'}}, ...
            'adaptStepSize',            {{'adaptStepSize', true}}, ...
            'adaptInterval',            {{'adaptInterval', 100}}, ...
            'adaptTarget',              {{'adaptTarget', 0.3}}, ...
            'adaptCovariance',          {{'adaptCovariance', true}}, ...
            'overdisp',                 {{'overdisp', 0.01}} ...
            )
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
        %% legacy identity
        function testDefaultsBitwiseLegacyLoop(testCase)
            gacelletest.assumeGPU(testCase);
            [y, mask, w, pars0, f, obj] = McmcSamplerInfraTest.r2starSetup();
            runSeed = 48463;

            rng(runSeed); parallel.gpu.rng(runSeed);
            out = mcmc().optimisation(y, mask, w, pars0, f, @obj.FWD, f.solver, f);
            testCase.verifyEqual(sort(fieldnames(out)), sort({'posterior';'mean';'std';'median'}));

            % reference: the legacy loop called directly, with the preprocessing of optimisation
            fd       = mcmc.check_set_default_basic(f);
            mask_idx = find(mask>0);
            yd       = utils.reshape_ND2GD(y, mask_idx);
            if ~ismatrix(w); wd = utils.reshape_ND2GD(w, mask_idx); elseif ~isempty(w); wd = w(:,mask_idx); else; wd = w; end
            p0       = utils.reshape_ND2GD_struct(pars0, mask);
            rng(runSeed); parallel.gpu.rng(runSeed);
            ref = mcmc.res2out(mcmc().metropolis_hastings(yd, p0, wd, fd, @obj.FWD, f.solver, f), fd, mask);
            testCase.verifyTrue(isequaln(out, ref), 'mcmc().optimisation differs from the legacy loop');
            testCase.verifyGreaterThan(numel(unique(out.posterior.R2star(:))), 1);   % the chain moved

            % every opt-in option given explicitly at its default: still the legacy path
            def = mcmc.sampler_infra_defaults();
            g   = f;
            for k = 1:size(def,1); g.(def{k,1}) = def{k,2}; end
            rng(runSeed); parallel.gpu.rng(runSeed);
            out2 = mcmc().optimisation(y, mask, w, pars0, g, @obj.FWD, f.solver, f);
            testCase.verifyTrue(isequaln(out2, ref), 'explicit defaults changed the output');
        end

        %% mcmc with the opt-in options == mcmc_bayes with the same options (same chain)
        function testOptionsMatchMcmcBayes(testCase, infraConfig)
            gacelletest.assumeGPU(testCase);
            [y, mask, w, pars0, f0, obj] = McmcSamplerInfraTest.r2starSetup();
            f = f0; f.iteration = 600; f.burnin = 0.5; f.thinning = 5;
            fn = fieldnames(infraConfig);
            for k = 1:numel(fn); f.(fn{k}) = infraConfig.(fn{k}); end
            runSeed = 48463;

            rng(runSeed); parallel.gpu.rng(runSeed);
            outA = mcmc().optimisation(y, mask, w, pars0, f, @obj.FWD, f0.solver, f0);
            rng(runSeed); parallel.gpu.rng(runSeed);
            outB = mcmc_bayes().optimisation(y, mask, w, pars0, f, @obj.FWD, f0.solver, f0);

            testCase.verifyEqual(sort(fieldnames(outA)), sort(fieldnames(outB)));
            for fld = {'posterior','mean','std','median','diagnostics'}
                testCase.verifyTrue(isequaln(outA.(fld{1}), outB.(fld{1})), ['out.' fld{1} ' differs between mcmc and mcmc_bayes']);
            end
            common = fieldnames(outA.settings);
            testCase.verifyEmpty(setdiff(common, fieldnames(outB.settings)));
            % overdispRule is descriptive text; mcmc_bayes's also covers unbounded hierarchical rows
            common = setdiff(common, {'overdispRule'});
            for k = 1:numel(common)
                testCase.verifyTrue(isequaln(outA.settings.(common{k}), outB.settings.(common{k})), ['out.settings.' common{k}]);
            end
            testCase.verifyGreaterThan(numel(unique(outA.posterior.R2star(:))), 1);
        end

        %% outputs of the opt-in path
        function testDiagnosticsOutput(testCase)
            gacelletest.assumeGPU(testCase);
            [y, mask, w, pars0, f, obj] = McmcSamplerInfraTest.r2starSetup();
            f.parameterTransform = 'sigmoid'; f.adaptStepSize = true; f.adaptCovariance = true;
            f.repetition = 2; f.overdisp = 0.1; f.iteration = 600; f.burnin = 0.5; f.thinning = 5;
            out = mcmc().optimisation(y, mask, w, pars0, f, @obj.FWD, f.solver, f);
            sz  = size(mask, 1:3);
            testCase.verifySize(out.diagnostics.acceptance, [sz 1 2]);
            testCase.verifyEqual(out.diagnostics.acceptanceBlocks, {'joint'});
            for p = {'M0','R2star','noise'}
                testCase.verifySize(out.diagnostics.stepSize.(p{1}), [sz 2]);
                testCase.verifySize(out.diagnostics.ess.(p{1}), sz);
                testCase.verifySize(out.diagnostics.rhat.(p{1}), sz);
                testCase.verifyTrue(all(isfinite(out.diagnostics.rhat.(p{1})(:))), p{1});
            end
            testCase.verifySize(out.diagnostics.adaptCovariance.proposalCov, [sz 3 3 2]);
            testCase.verifyEqual(out.settings.parameterTransform, {'sigmoid','sigmoid','sigmoid'});
            testCase.verifyTrue(all(out.posterior.M0(:) >= f.lb(1) & out.posterior.M0(:) <= f.ub(1)));
            % fewer than 4 kept samples per chain: ESS and R-hat are NaN, no error
            f.iteration = 20; f.burnin = 10; f.thinning = 5; f.adaptCovariance = false;
            out = mcmc().optimisation(y, mask, w, pars0, f, @obj.FWD, f.solver, f);
            testCase.verifyTrue(all(isnan(out.diagnostics.ess.R2star(:))) && all(isnan(out.diagnostics.rhat.R2star(:))));
        end

        %% option detection and errors (no GPU)
        function testUseSamplerInfra(testCase, nonDefaultOption)
            testCase.verifyFalse(mcmc.use_sampler_infra(struct()));
            testCase.verifyFalse(mcmc.use_sampler_infra([]));
            testCase.verifyFalse(mcmc.use_sampler_infra(struct('iteration', 100, 'algorithm', 'MH', 'likelihood', 'marginal_noise')));
            def = mcmc.sampler_infra_defaults();
            f   = struct();
            for k = 1:size(def,1); f.(def{k,1}) = def{k,2}; end
            testCase.verifyFalse(mcmc.use_sampler_infra(f));
            f.(nonDefaultOption{1}) = nonDefaultOption{2};
            testCase.verifyTrue(mcmc.use_sampler_infra(f));
            testCase.verifyEqual(mcmc.nondefault_options(f, def), nonDefaultOption(1));
        end

        function testOptionErrors(testCase)
            run = @(f) mcmc().optimisation([], [], [], [], f, []);
            testCase.verifyError(@() run(struct('algorithm', 'GW', 'parameterTransform', 'sigmoid')), 'mcmc:unsupportedAlgorithm');
            testCase.verifyError(@() run(struct('updateScheme', 'blockwise')), 'mcmc:invalidUpdateScheme');
            testCase.verifyError(@() run(struct('adaptCovariance', true)), 'mcmc:adaptCovariance');
            testCase.verifyError(@() run(struct('adaptCovariance', true, 'adaptStepSize', true, 'updateScheme', 'componentwise')), 'mcmc:adaptCovariance');
            testCase.verifyError(@() run(struct('adaptCovariance', 2, 'adaptStepSize', true)), 'mcmc:adaptCovariance');
            f = mcmc.check_set_default_infra(struct('updateScheme', 'componentwise', 'overdisp', []));
            testCase.verifyEqual([f.adaptTarget f.overdisp], [0.44 0]);
            f = struct('modelParams', {{'a'}}, 'lb', 0, 'ub', 1, 'xStepSize', 0.1, 'parameterTransform', 'sigmoid');
            testCase.verifyError(@() mcmc().metropolis_hastings_adaptive(zeros(3,2), struct('a', [0.5 0.5]), [], f, @(p) p.a), 'mcmc:noNoise');
        end

        function testForwardSizeCheck(testCase)
            testCase.verifyError(@() mcmc.check_forward_size(zeros(1,5), 16, 5), 'mcmc:forwardSize');
            testCase.verifyError(@() mcmc.check_forward_size(zeros(16,4), 16, 5), 'mcmc:forwardSize');
            testCase.verifyError(@() mcmc.check_forward_size(zeros(16,5,2), 16, 5), 'mcmc:forwardSize');
            mcmc.check_forward_size(zeros(16,5), 16, 5);         % no error
            % end to end: 3-D single-slice data [nx,ny,Nte] is read as a volume with one measurement
            gacelletest.assumeGPU(testCase);
            te  = linspace(0, 0.04, 6).';
            fwd = @(p) p.M0 .* exp(-te .* p.R2star);
            f   = struct('modelParams', {{'M0','R2star','noise'}}, 'lb', [0 0.1 0.001], 'ub', [2 200 0.1], ...
                         'xStepSize', [0.01 1 0.005], 'algorithm', 'MH', 'parameterTransform', 'sigmoid', ...
                         'iteration', 20, 'burnin', 10, 'thinning', 1, 'metric', {{'mean'}});
            x0  = struct('M0', ones(3,4), 'R2star', 30*ones(3,4), 'noise', 0.05*ones(3,4));
            y3  = rand(3, 4, numel(te)) + 0.5;
            testCase.verifyError(@() mcmc().optimisation(y3, true(3,4), [], x0, f, fwd), 'mcmc:forwardSize');
            out = mcmc().optimisation(reshape(y3, [3 4 1 numel(te)]), true(3,4), [], x0, f, fwd);
            testCase.verifyTrue(all(isfinite(out.mean.R2star(:))));
        end

        %% helpers (moved from McmcBayesUnitTest with the helpers)
        function testTransformRoundTrip(testCase, transformMethod)
            [lb, ub] = McmcSamplerInfraTest.boundsFor(transformMethod);
            x = linspace(lb + 0.01*(ub-lb), ub - 0.01*(ub-lb), 101);

            % double
            u  = mcmc.transform_forward(x, transformMethod, lb, ub);
            x2 = mcmc.transform_inverse(u, transformMethod, lb, ub);
            testCase.verifyLessThanOrEqual(max(abs(x2 - x)), 1e-10*(ub-lb));

            % single
            xs  = single(x);
            us  = mcmc.transform_forward(xs, transformMethod, single(lb), single(ub));
            xs2 = mcmc.transform_inverse(us, transformMethod, single(lb), single(ub));
            testCase.verifyClass(xs2, 'single');
            testCase.verifyLessThanOrEqual(double(max(abs(xs2 - xs))), 1e-4*(ub-lb));
        end

        function testTransformLogJacobianFiniteDifference(testCase, transformMethod)
            [lb, ub] = McmcSamplerInfraTest.boundsFor(transformMethod);
            switch transformMethod
                case 'log';     u = linspace(log(lb)+0.1, log(ub)-0.1, 41);
                case 'sigmoid'; u = linspace(-8, 8, 41);
                otherwise;      u = linspace(lb, ub, 41);
            end
            h    = 1e-5;
            fd   = (mcmc.transform_inverse(u+h, transformMethod, lb, ub) - ...
                    mcmc.transform_inverse(u-h, transformMethod, lb, ub)) ./ (2*h);
            logJ = mcmc.transform_logjac(u, transformMethod, lb, ub);
            testCase.verifyLessThanOrEqual(max(abs(logJ - log(abs(fd)))), 1e-6);
        end

        function testTransformStableAtLargeU(testCase)
            lb = 0.5; ub = 2;
            u  = [-1e4 -1e3 -100 -50 50 100 1e3 1e4];
            for cls = {'double','single'}
                uc   = cast(u, cls{1});
                x    = mcmc.transform_inverse(uc, 'sigmoid', cast(lb,cls{1}), cast(ub,cls{1}));
                logJ = mcmc.transform_logjac(uc, 'sigmoid', cast(lb,cls{1}), cast(ub,cls{1}));
                testCase.verifyTrue(all(isfinite(x)), ['sigmoid inverse not finite, ' cls{1}]);
                testCase.verifyTrue(all(x >= lb & x <= ub), ['sigmoid inverse outside box, ' cls{1}]);
                testCase.verifyTrue(all(isfinite(logJ)), ['sigmoid logjac not finite, ' cls{1}]);
                testCase.verifyLessThanOrEqual(max(abs(double(logJ) - (log(ub-lb) - abs(u))) ./ abs(u)), 1e-6);

                % log: the Jacobian is u itself, finite for any finite u
                logJ = mcmc.transform_logjac(uc, 'log', cast(0,cls{1}), cast(1,cls{1}));
                testCase.verifyEqual(logJ, uc);
            end

            % forward transform at (and beyond) the bounds is finite because of the eps clamp
            for m = {'sigmoid','log'}
                lb = 0; ub = 0.1;
                u  = mcmc.transform_forward([lb-1 lb ub ub+1], m{1}, lb, ub);
                testCase.verifyTrue(all(isfinite(u)), ['forward not finite at bounds, ' m{1}]);
                u  = mcmc.transform_forward(single([lb ub]), m{1}, single(lb), single(ub));
                testCase.verifyTrue(all(isfinite(u)), ['forward not finite at bounds (single), ' m{1}]);
            end
        end

        function testTransformPerParameterCell(testCase)
            % string applies to all parameters
            testCase.verifyEqual(mcmc.parse_transform('Sigmoid', 3), {'sigmoid','sigmoid','sigmoid'});
            testCase.verifyEqual(mcmc.parse_transform("log", 2), {'log','log'});
            % cell: one entry per parameter, case-insensitive
            testCase.verifyEqual(mcmc.parse_transform({'linear','Log','SIGMOID'}, 3), {'linear','log','sigmoid'});
            % wrong length or unknown name
            testCase.verifyError(@() mcmc.parse_transform({'linear','log'}, 3), 'mcmc:invalidTransform');
            testCase.verifyError(@() mcmc.parse_transform('logit', 3), 'mcmc:invalidTransform');
            testCase.verifyError(@() mcmc.parse_transform(1, 3), 'mcmc:invalidTransform');

            % bounds
            testCase.verifyError(@() mcmc.check_transform_bounds({'log'}, -1, 1), 'mcmc:invalidBounds');
            testCase.verifyError(@() mcmc.check_transform_bounds({'sigmoid'}, 0, Inf), 'mcmc:invalidBounds');
            mcmc.check_transform_bounds({'linear','log','sigmoid'}, [-Inf 0 0], [Inf 1 1]);    % no error

            % rows are transformed independently with their own method and bounds
            method  = {'linear','sigmoid','log'};
            lb      = [0; 0.5; 0.001];
            ub      = [1; 2;   200];
            x       = [0.3 0.7; 1.2 0.9; 30 0.05];
            u       = mcmc.transform_forward(x, method, lb, ub);
            for k = 1:3
                testCase.verifyEqual(u(k,:), mcmc.transform_forward(x(k,:), method{k}, lb(k), ub(k)));
            end
            logJ    = mcmc.transform_logjac(u, method, lb, ub);
            testCase.verifyEqual(logJ(1,:), [0 0]);
            testCase.verifyEqual(logJ(3,:), u(3,:));
            testCase.verifyEqual(mcmc.transform_inverse(u, method, lb, ub), x, 'AbsTol', 1e-10);
        end

        function testFusedKernelMatchesHelpers(testCase)
            % the sampler uses a fused GPU kernel; it must match the reference helpers
            gacelletest.assumeGPU(testCase);
            method  = {'linear','sigmoid','log'};
            lb      = gpuArray(single([0; 0.5; 0.001]));
            ub      = gpuArray(single([2; 2;   200]));
            u       = gpuArray(single([linspace(0.1,1.9,201); linspace(-60,60,201); linspace(-6,5,201)]));
            code    = gpuArray(single(mcmc.transform_code(method)));
            [x, logJ] = mcmc.transform_inverse_logjac_fused(u, code, lb, ub);
            xRef    = mcmc.transform_inverse(u, method, lb, ub);
            logJRef = mcmc.transform_logjac(u, method, lb, ub);
            testCase.verifyClass(gather(x), 'single');
            testCase.verifyEqual(gather(x),    gather(xRef),    'RelTol', single(1e-5), 'AbsTol', single(1e-6));
            testCase.verifyEqual(gather(logJ), gather(logJRef), 'RelTol', single(1e-5), 'AbsTol', single(1e-5));
        end

        function testEssRhatIid(testCase)
            rng(20260926);
            Nv = 400; N = 2000; M = 4;
            x  = randn(Nv, N, M);
            neff = mcmc.ess(x);
            R    = mcmc.rhat(x);
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
            neff  = mcmc.ess(x);
            ratio = median(neff) / essTrue;
            testCase.verifyGreaterThanOrEqual(ratio, 0.9, sprintf('AR(1) ESS ratio %.3f', ratio));
            testCase.verifyLessThanOrEqual(ratio, 1.1, sprintf('AR(1) ESS ratio %.3f', ratio));
            testCase.verifyLessThanOrEqual(median(mcmc.rhat(x)), 1.01);
        end

        function testRhatDetectsNonMixing(testCase)
            rng(20260928);
            Nv = 100; N = 1000; M = 4;
            x  = randn(Nv, N, M) + reshape(0:M-1, 1, 1, M);  % chain m shifted by (m-1) SD
            testCase.verifyGreaterThanOrEqual(min(mcmc.rhat(x)), 1.1);
        end

        function testAcovCholBatch(testCase)
            rng(61);
            devs = {@(x) x};
            if canUseGPU; devs{end+1} = @gpuArray; end
            for kd = 1:numel(devs)
                for d = 1:8
                    N = 50; C = zeros(d, d, N); Lref = zeros(d, d, N);
                    for n = 1:N
                        B = randn(d, d+2); C(:,:,n) = B*B.'/(d+2) + 1e-3*eye(d) .* 10.^(2*randn);
                        Lref(:,:,n) = chol(C(:,:,n), 'lower');
                    end
                    [L, ok] = mcmc.chol_batch(devs{kd}(C));
                    L = gather(L);
                    testCase.verifyTrue(all(gather(ok)));
                    testCase.verifyLessThanOrEqual(max(abs(L - Lref), [], 'all'), 1e-10 * max(abs(Lref), [], 'all'), sprintf('d = %d', d));
                    testCase.verifyEqual(triu(L(:,:,1), 1), zeros(d));      % lower triangular
                end
                % indefinite and NaN pages are flagged, the others are not affected
                C = repmat(eye(3), 1, 1, 3); C(:,:,2) = [1 2 0; 2 1 0; 0 0 1]; C(1,1,3) = NaN;
                [L, ok] = mcmc.chol_batch(devs{kd}(C));
                testCase.verifyEqual(gather(ok), [true false false]);
                testCase.verifyEqual(gather(L(:,:,1)), eye(3));
            end
        end

        function testAcovWelford(testCase)
            rng(62); d = 4; Nv = 30; T = 500;
            A = randn(d); X = 3 + pagemtimes(A, randn(d, T, Nv)) .* reshape(logspace(-3, 2, Nv), 1, 1, Nv);   % [d,T,Nv]
            m = zeros(d, Nv); M2 = zeros(d, d, Nv); n = 0;
            for t = 1:T
                [m, M2, n] = mcmc.welford_update(m, M2, n, squeeze(X(:,t,:)));
            end
            testCase.verifyEqual(n, T);
            for v = 1:Nv
                Cref = cov(X(:,:,v).');
                testCase.verifyEqual(m(:,v), mean(X(:,:,v), 2), 'RelTol', 1e-10);
                testCase.verifyEqual(M2(:,:,v)/(n-1), Cref, 'AbsTol', 1e-10*max(diag(Cref)));
            end
            % single-precision GPU input accumulated in double on the GPU
            if canUseGPU
                mG = zeros(d, Nv, 'double', 'gpuArray'); M2G = zeros(d, d, Nv, 'double', 'gpuArray'); nG = 0;
                Xs = single(X);
                for t = 1:T; [mG, M2G, nG] = mcmc.welford_update(mG, M2G, nG, gpuArray(squeeze(Xs(:,t,:)))); end
                testCase.verifyTrue(isa(M2G, 'gpuArray') && strcmp(underlyingType(M2G), 'double'));
                Cref = cov(double(Xs(:,:,7)).');
                testCase.verifyEqual(gather(M2G(:,:,7))/(nG-1), Cref, 'AbsTol', 1e-10*max(diag(Cref)));
            end
        end

        function testAcovRefreshFallback(testCase)
            d = 2; N = 3; n = 50;
            C = cat(3, [2 0.5; 0.5 1], zeros(2), [1 0.9; 0.9 1]);
            M2 = C * (n-1);
            accN  = [10 10 2];                                  % voxel 3: < d+1 acceptances
            Lprev = repmat(single([0.3 0; 0 0.4]), 1, 1, N);
            [L, valid] = mcmc.acov_refresh(M2, n, accN, Lprev, false(1, N), 1e-6, 1e-12);
            Creg = C(:,:,1) + 1e-6*diag(diag(C(:,:,1))) + (1e-12*2 + realmin('double'))*eye(2);
            testCase.verifyEqual(double(L(:,:,1)), chol(Creg, 'lower'), 'RelTol', 1e-6);   % single output
            testCase.verifyEqual(L(:,:,2), Lprev(:,:,2));        % zero variance: previous factor kept
            testCase.verifyEqual(L(:,:,3), Lprev(:,:,3));        % too few acceptances: previous factor kept
            testCase.verifyEqual(valid, [true false false]);
            [~, valid] = mcmc.acov_refresh(M2, n, accN, Lprev, [false true false], 1e-6, 1e-12);
            testCase.verifyEqual(valid, [true true false]);      % once valid, stays valid (older factor kept)
            % the proposal increment is L*z per voxel
            Lp = randn(3, 3, 4); z = randn(3, 4);
            s  = mcmc.acov_step(Lp, z);
            for v = 1:4; testCase.verifyEqual(s(:,v), Lp(:,:,v)*z(:,v), 'AbsTol', 1e-12); end
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
    end
end
