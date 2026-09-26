classdef McmcBayesUnitTest < matlab.unittest.TestCase
    % Unit tests for the EXPERIMENTAL mcmc_bayes subclass.
    % Phase 0: legacy-option detection (mcmc_bayes.isLegacy) and the
    % not-implemented guard for Phase 2+ options.
    % Phase 1: parameter transforms, per-parameter parsing, R-hat/ESS and
    % a small GPU run of the new sampling path.
    % Phase 2: marginal likelihoods vs numerical integration, weighted form,
    % parameter dropping/restoring, S0Param errors, nuisance draws, GPU runs.
    %
    % All tests except testNewPath*, testFusedKernel* and testRecoverNuisance*
    % are pure math and need no GPU.
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
    %   Phase 2 (double precision, CPU):
    %   marginal logL difference between two theta vs numerical integration
    %     (integral / integral2, RelTol 1e-10) of the joint density     : |diff| <= 1e-6
    %     Zellner prior with k = 1e12 (broad-limit error (m/2)(y'Wg)^2/(k c RSS) ~ 1e-8)
    %     flat prior U(-1e3, 1e3) on S0 (truncation error negligible)
    %   Zellner vs flat difference for the same theta pair            : >= 0.1 (test is discriminative)
    %   weighted form vs unweighted on W^(1/2)-scaled y and g          : AbsTol 1e-9
    %   W = I vs reference unweighted formula (y'y - (y'g)^2/g'g)       : AbsTol 1e-9
    %   scale invariance g -> 3g: Zellner unchanged, flat -log(3)       : AbsTol 1e-9
    %   degenerate states (g = 0, RSS = 0, NaN)                         : logL == -Inf (never NaN)
    %   Phase 2 (GPU, single): post-hoc draws, 2e5 samples of one state: mean(sigma^2), mean(S0),
    %     var(S0) within 5 MC standard errors of the analytic value (InvGamma / Student-t moments)
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

        % Phase 3+ options, still not implemented
        phase3Option = struct( ...
            'prior',                    {{'prior', struct('hierarchical', struct())}} ...
            )

        marginalLikelihood = {'marginal_S0noise','marginal_S0noise_flat'}

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

        function testPhase3OptionErrorsNotImplemented(testCase, phase3Option)
            % the guard fires before any GPU code, so no GPU is needed
            fitting = struct();
            fitting.(phase3Option{1}) = phase3Option{2};
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

        %% Phase 2: marginal likelihoods (pure math, double precision)
        function testMarginalNoiseVsNumericalIntegration(testCase)
            [y, w, g1, g2] = McmcBayesUnitTest.marginalVoxel();
            y  = y ./ 1.05;                     % normalised data, no amplitude
            m  = numel(y);
            % numerical: log int (2 pi s2)^(-m/2) exp(-r'Wr/(2 s2)) (1/s2) ds2, with s2 = exp(t)
            num = zeros(1,2);
            G   = {g1, g2};
            for k = 1:2
                R    = sum(w.*(y - G{k}).^2);
                logf = @(t) -m/2*log(2*pi) - m/2*t - R./(2*exp(t));     % (1/s2) ds2 = dt
                t0   = log(R/m); K = logf(t0);
                num(k) = log(integral(@(t) exp(logf(t) - K), t0-60, t0+60, 'RelTol', 1e-10, 'AbsTol', 0)) + K;
            end
            ana = mcmc_bayes.loglik_marginal([g1 g2], [y y], [w w], 'marginal_noise', 0);
            testCase.verifyEqual(ana(1)-ana(2), num(1)-num(2), 'AbsTol', 1e-6);
            % the full analytic constant too: log L = log Gamma(m/2) - (m/2) log(pi) - (m/2) log R
            testCase.verifyEqual(ana(1) + gammaln(m/2) - m/2*log(pi), num(1), 'AbsTol', 1e-6);
        end

        function testMarginalS0noiseVsNumericalIntegration(testCase, marginalLikelihood)
            [y, w, g1, g2] = McmcBayesUnitTest.marginalVoxel();
            num = [McmcBayesUnitTest.logMarginal2D(y, w, g1, marginalLikelihood), ...
                   McmcBayesUnitTest.logMarginal2D(y, w, g2, marginalLikelihood)];
            ana = mcmc_bayes.loglik_marginal([g1 g2], [y y], [w w], marginalLikelihood, 0);
            testCase.verifyEqual(ana(1)-ana(2), num(1)-num(2), 'AbsTol', 1e-6, ...
                sprintf('%s: analytic %.8f vs numerical %.8f', marginalLikelihood, ana(1)-ana(2), num(1)-num(2)));
        end

        function testZellnerAndFlatDiffer(testCase)
            % the two S0 priors give different theta-dependence (so the tests above can tell them apart)
            [y, w, g1, g2] = McmcBayesUnitTest.marginalVoxel();
            z = mcmc_bayes.loglik_marginal([g1 g2], [y y], [w w], 'marginal_S0noise', 0);
            f = mcmc_bayes.loglik_marginal([g1 g2], [y y], [w w], 'marginal_S0noise_flat', 0);
            testCase.verifyGreaterThanOrEqual(abs((z(1)-z(2)) - (f(1)-f(2))), 0.1);
        end

        function testMarginalWeightedForm(testCase)
            rng(11);
            m = 12; Nv = 5;
            g = exp(-rand(m,Nv)*3); y = (0.8+0.4*rand(1,Nv)).*g + 0.02*randn(m,Nv);
            w = 0.2 + rand(m,Nv);
            for name = {'marginal_noise','marginal_S0noise','marginal_S0noise_flat'}
                % weighted form == unweighted form on W^(1/2)-scaled data and model
                [lw, sw] = mcmc_bayes.loglik_marginal(g, y, w, name{1}, 0);
                [lu, su] = mcmc_bayes.loglik_marginal(sqrt(w).*g, sqrt(w).*y, ones(m,Nv), name{1}, 0);
                testCase.verifyEqual(lw, lu, 'AbsTol', 1e-9, name{1});
                testCase.verifyEqual(sw(1,:), su(1,:), 'RelTol', 1e-9, name{1});
            end

            % W = I: the reference (BayesIVIM) unweighted formula
            one = ones(m,Nv);
            ref = -m/2*log(sum(y.^2) - sum(y.*g).^2./sum(g.^2));
            testCase.verifyEqual(mcmc_bayes.loglik_marginal(g, y, one, 'marginal_S0noise', 0), ref, 'AbsTol', 1e-9);

            % cached statistics: RSS (residual form) == y'Wy - (y'Wg)^2/g'Wg, Shat, g'Wg
            [~, st] = mcmc_bayes.loglik_marginal(g, y, w, 'marginal_S0noise', 0);
            yWg = sum(w.*y.*g); gWg = sum(w.*g.^2);
            testCase.verifyEqual(st(1,:), sum(w.*y.^2) - yWg.^2./gWg, 'RelTol', 1e-8);
            testCase.verifyEqual(st(2,:), yWg./gWg, 'RelTol', 1e-12);
            testCase.verifyEqual(st(3,:), gWg, 'RelTol', 1e-12);

            % scale invariance g -> 3g: Zellner broad limit unchanged, flat shifts by -log(3)
            z1 = mcmc_bayes.loglik_marginal(g, y, w, 'marginal_S0noise', 0);
            z3 = mcmc_bayes.loglik_marginal(3*g, y, w, 'marginal_S0noise', 0);
            testCase.verifyEqual(z3, z1, 'AbsTol', 1e-9);
            f1 = mcmc_bayes.loglik_marginal(g, y, w, 'marginal_S0noise_flat', 0);
            f3 = mcmc_bayes.loglik_marginal(3*g, y, w, 'marginal_S0noise_flat', 0);
            testCase.verifyEqual(f3, f1 - log(3), 'AbsTol', 1e-9);
        end

        function testMarginalDegenerateStatesRejected(testCase)
            m = 6;
            y = [1; 0.8; 0.6; 0.5; 0.4; 0.3];
            w = ones(m,1);
            g0   = zeros(m,1);              % g'Wg = 0
            gFit = y/2;                     % y = 2*g exactly -> RSS = 0
            gNaN = [NaN; y(2:end)];
            gInf = [Inf; y(2:end)];
            for name = {'marginal_S0noise','marginal_S0noise_flat'}
                l = mcmc_bayes.loglik_marginal([g0 gFit gNaN gInf], repmat(y,1,4), repmat(w,1,4), name{1}, 1e-12);
                testCase.verifyEqual(l, -Inf(1,4), name{1});
            end
            l = mcmc_bayes.loglik_marginal([y gNaN], [y y], [w w], 'marginal_noise', 1e-12);
            testCase.verifyEqual(l, -Inf(1,2));
            % single precision keeps its class
            l = mcmc_bayes.loglik_marginal(single([g0 y]), single([y y]), single([w w]), 'marginal_S0noise', single(1e-12));
            testCase.verifyClass(l, 'single');
            testCase.verifyEqual(l(1), single(-Inf));
        end

        %% Phase 2: parameter dropping/restoring and errors (pure, no GPU)
        function testSetupLikelihoodDropsAndRestores(testCase)
            fitting.modelParams = {'M0';'R2star';'noise'};
            fitting.lb          = [0; 0.1; 0.001];
            fitting.ub          = [2; 200; 0.1];
            fitting.xStepSize   = [0.01; 1; 0.005];
            fitting.parameterTransform = {'linear','log','sigmoid'};
            fitting.likelihood  = 'marginal_S0noise';
            fitting.S0Param     = 'M0';
            [fs, lik] = mcmc_bayes.setup_likelihood(fitting);
            testCase.verifyEqual(fs.modelParams, {'R2star'});
            testCase.verifyEqual([fs.lb fs.ub fs.xStepSize], [0.1 200 1]);
            testCase.verifyEqual(fs.parameterTransform, {'log'});
            testCase.verifyEqual(lik.droppedParams, {'M0','noise'});
            testCase.verifyEqual(lik.fixedParams, {'M0','noise'});
            testCase.verifyEqual(lik.fittingOut.modelParams, {'M0';'R2star';'noise'});
            testCase.verifyEqual(lik.fittingOut.lb, fitting.lb);
            testCase.verifyEqual(lik.shapeOffset, 0);

            % case-insensitive name, flat variant
            fitting.likelihood = 'Marginal_S0noise_FLAT';
            [~, lik] = mcmc_bayes.setup_likelihood(fitting);
            testCase.verifyEqual(lik.name, 'marginal_S0noise_flat');
            testCase.verifyEqual(lik.shapeOffset, 1);

            % marginal_noise: noise dropped, M0 kept (order kept)
            f2 = rmfield(fitting,'S0Param'); f2.likelihood = 'marginal_noise';
            fs = mcmc_bayes.setup_likelihood(f2);
            testCase.verifyEqual(fs.modelParams, {'M0';'R2star'});
            testCase.verifyEqual(fs.parameterTransform, {'linear','log'});

            % marginal_noise without a noise parameter: nothing dropped, 'noise' appended to the output
            f3 = struct('modelParams', {{'fa','Da'}}, 'lb', [0 0], 'ub', [1 3], 'xStepSize', [0.1 0.1], 'likelihood', 'marginal_noise');
            [fs, lik] = mcmc_bayes.setup_likelihood(f3);
            testCase.verifyEqual(fs.modelParams, {'fa';'Da'});
            testCase.verifyEqual(lik.fittingOut.modelParams, {'fa';'Da';'noise'});
            testCase.verifyEqual(lik.fittingOut.ub(3), Inf);

            % gaussian: nothing dropped
            f4 = rmfield(fitting,{'likelihood','S0Param'});
            [fs, lik] = mcmc_bayes.setup_likelihood(f4);
            testCase.verifyEqual(fs.modelParams, {'M0';'R2star';'noise'});
            testCase.verifyFalse(lik.isMarginal);
        end

        function testSetupLikelihoodErrors(testCase)
            base.modelParams = {'M0';'R2star';'noise'};
            base.lb          = [0; 0.1; 0.001];
            base.ub          = [2; 200; 0.1];
            base.xStepSize   = [0.01; 1; 0.005];

            % errors are raised by optimisation before any GPU code
            run = @(f) mcmc_bayes().optimisation([], [], [], [], f, []);

            f = base; f.likelihood = 'marginal_S0noise';                    % missing S0Param
            testCase.verifyError(@() run(f), 'mcmc_bayes:S0Param');
            f.S0Param = 'S0';                                               % not a modelParam
            testCase.verifyError(@() run(f), 'mcmc_bayes:S0Param');
            f.S0Param = 'noise';
            testCase.verifyError(@() run(f), 'mcmc_bayes:S0Param');
            f = base; f.S0Param = 'M0';                                     % S0Param with gaussian
            testCase.verifyError(@() run(f), 'mcmc_bayes:S0Param');
            f.likelihood = 'marginal_noise';                                % S0Param with marginal_noise
            testCase.verifyError(@() run(f), 'mcmc_bayes:S0Param');
            f = base; f.likelihood = 'orton';
            testCase.verifyError(@() run(f), 'mcmc_bayes:invalidLikelihood');
            f = base; f.likelihood = 'marginal_S0noise'; f.S0Param = 'M0';
            f.parameterTransform = {'log'};                                 % not one per modelParams
            testCase.verifyError(@() run(f), 'mcmc_bayes:invalidTransform');
            f = rmfield(f,'parameterTransform'); f.xStepSize = [1 1];
            testCase.verifyError(@() run(f), 'mcmc_bayes:xStepSize');
            f = base; f.modelParams = {'noise'}; f.lb = 0; f.ub = 1; f.xStepSize = 1; f.likelihood = 'marginal_noise';
            testCase.verifyError(@() run(f), 'mcmc_bayes:noSampledParams');
            f = base; f.modelParams = {'M0';'R2star';'sigma'};              % gaussian needs 'noise'
            f.updateScheme = 'componentwise';                               % (non-default -> new path)
            testCase.verifyError(@() run(f), 'mcmc_bayes:noNoise');
        end

        %% Phase 2: post-hoc nuisance draws (GPU randg)
        function testRecoverNuisance(testCase)
            gacelletest.assumeGPU(testCase);
            fitting.modelParams = {'M0';'R2star';'noise'};
            fitting.lb = [0; 0.1; 0.001]; fitting.ub = [2; 200; 0.1]; fitting.xStepSize = [0.01; 1; 0.005];
            fitting.likelihood = 'marginal_S0noise'; fitting.S0Param = 'M0';
            [~, lik] = mcmc_bayes.setup_likelihood(fitting);

            m = 12; RSS = 0.005; Shat = 1.03; c = 4.2;
            Nv = 2; Ns = 1e5; Nrep = 2;
            statsPost = repmat(single([RSS; Shat; c]), 1, Nv, Ns, Nrep);
            xPost.R2star = rand(Nv, Ns, Nrep, 'single');
            parallel.gpu.rng(5);
            xOut = mcmc_bayes.recover_nuisance(xPost, statsPost, lik, m);

            % full modelParams order, sampled field untouched, sizes
            testCase.verifyEqual(fieldnames(xOut), {'M0';'R2star';'noise'});
            testCase.verifyEqual(xOut.R2star, xPost.R2star);
            testCase.verifyEqual(size(xOut.M0), [Nv Ns Nrep]);
            testCase.verifyEqual(size(xOut.noise), [Nv Ns Nrep]);
            testCase.verifyTrue(all(xOut.noise(:) > 0));

            % moments: sigma^2 ~ InvGamma(a, RSS/2), S0 ~ Shat + t_{2a} * sqrt(RSS/(2a c))
            a   = m/2; b = RSS/2; N = Nv*Ns*Nrep;
            s2  = double(xOut.noise(:)).^2;
            Es2 = b/(a-1); Vs2 = b^2/((a-1)^2*(a-2));
            testCase.verifyLessThanOrEqual(abs(mean(s2) - Es2), 5*sqrt(Vs2/N));
            S0  = double(xOut.M0(:));
            VS0 = Es2/c;                                        % var of the t mixture = E[sigma^2]/c
            testCase.verifyLessThanOrEqual(abs(mean(S0) - Shat), 5*sqrt(VS0/N));
            k4  = 3*(2*a-2)/(2*a-4);                            % kurtosis of t_{2a}
            testCase.verifyLessThanOrEqual(abs(var(S0) - VS0), 5*VS0*sqrt((k4-1)/N));
        end

        function testNewPathMarginalRuns(testCase, marginalLikelihood)
            gacelletest.assumeGPU(testCase);
            [y, mask, w, pars0, fitting, obj] = McmcBayesUnitTest.r2starSetup();
            f = fitting;
            f.likelihood    = marginalLikelihood;
            f.S0Param       = 'M0';
            f.metric        = {'mean','std','median','mode'};
            f.parameterTransform = {'linear','sigmoid','log'};      % one per FULL modelParams
            f.repetition    = 2;
            f.overdisp      = 0.01;
            out = mcmc_bayes().optimisation(y, mask, w, pars0, f, @obj.FWD, 'mcmc', f);

            testCase.verifyEqual(fieldnames(out.posterior), {'M0';'R2star';'noise'});
            sz = [nnz(mask) numel(21:2:200) 2];     % [Nv, Ns, Nrep], Nburnin = 20, thinning = 2
            for p = {'M0','R2star','noise'}
                testCase.verifyEqual(size(out.posterior.(p{1})), sz);
                for metric = {'mean','std','median','mode'}
                    testCase.verifyTrue(all(isfinite(out.(metric{1}).(p{1})(:))), [metric{1} '.' p{1}]);
                end
            end
            testCase.verifyGreaterThanOrEqual(min(out.posterior.R2star(:)), single(fitting.lb(2)));
            testCase.verifyLessThanOrEqual(max(out.posterior.R2star(:)), single(fitting.ub(2)));
            testCase.verifyTrue(all(out.posterior.noise(:) > 0));
            testCase.verifyEqual(fieldnames(out.diagnostics.stepSize), {'R2star'});
            testCase.verifyTrue(isfield(out.diagnostics.rhat, 'M0'));
            testCase.verifyEqual(out.settings.likelihood, marginalLikelihood);
            testCase.verifyEqual(out.settings.sampledParams, {'R2star'});
            testCase.verifyEqual(out.settings.droppedParams, {'M0','noise'});
            testCase.verifyEqual(out.settings.nuisance.fields, {'noise','M0'});
            % posterior mean M0 near the truth range (1 to 1.1), noise near 1/50 (loose sanity bounds)
            testCase.verifyTrue(all(abs(out.mean.M0(:) - 1.05) < 0.2));
            testCase.verifyTrue(all(out.mean.noise(:) > 0.005 & out.mean.noise(:) < 0.08));
        end

        function testNewPathMarginalNoiseRuns(testCase)
            gacelletest.assumeGPU(testCase);
            [y, mask, w, pars0, fitting, obj] = McmcBayesUnitTest.r2starSetup();
            f = fitting;
            f.likelihood    = 'marginal_noise';
            f.updateScheme  = 'componentwise';
            out = mcmc_bayes().optimisation(y, mask, w, pars0, f, @obj.FWD, 'mcmc', f);
            testCase.verifyEqual(fieldnames(out.posterior), {'M0';'R2star';'noise'});
            testCase.verifyEqual(out.settings.sampledParams, {'M0','R2star'});
            testCase.verifyEqual(size(out.diagnostics.acceptance), [size(mask,1:3) 2]);
            testCase.verifyTrue(all(out.posterior.noise(:) > 0));
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

        % one synthetic monoexponential voxel and two theta values (amplitude-free g)
        function [y, w, g1, g2] = marginalVoxel()
            rng(20260926);
            te  = linspace(0, 40e-3, 12).';
            y   = 1.05*exp(-te*32) + 0.02*randn(size(te));
            w   = 0.5 + rand(size(te));
            g1  = exp(-te*30);
            g2  = exp(-te*45);
        end

        % log marginal likelihood by 2D numerical integration over S0 and sigma^2,
        % directly from the joint density (no analytic S0 or sigma^2 integral used)
        %   'marginal_S0noise'      : S0 | sigma^2 ~ N(0, k sigma^2/g'Wg), k = 1e12 (broad Zellner), p(sigma^2) ∝ 1/sigma^2
        %   'marginal_S0noise_flat' : S0 ~ U(-A, A), A = 1e3,                                        p(sigma^2) ∝ 1/sigma^2
        % Variables: t = log sigma^2, S0 = S0c + z*sqrt(sigma^2/c) (the substitution only centres and
        % scales the quadrature; S0c and c are ordinary numbers here, the Jacobian is included).
        function logL = logMarginal2D(y, w, g, name)
            m   = numel(y);
            c   = sum(w.*g.^2);
            S0c = sum(w.*y.*g)/c;
            R0  = sum(w.*(y - S0c*g).^2);
            k   = 1e12; A = 1e3;
            t0  = log(R0/m);
            logf = @(t, z) McmcBayesUnitTest.logJoint(t, z, y, w, g, c, S0c, m, name, k, A);
            K   = logf(t0, 0);
            val = integral2(@(t,z) exp(logf(t,z) - K), t0-40, t0+40, -60, 60, 'RelTol', 1e-10, 'AbsTol', 0, 'Method', 'iterated');
            logL = log(val) + K;
        end

        function lf = logJoint(t, z, y, w, g, c, S0c, m, name, k, A)
            s2  = exp(t);
            S0  = S0c + z.*sqrt(s2./c);
            % Q = sum_i w_i (y_i - S0 g_i)^2, expanded elementwise over the (t,z) arrays
            Q   = sum(w.*y.^2) - 2*S0.*sum(w.*y.*g) + S0.^2.*sum(w.*g.^2);
            lf  = -m/2*log(2*pi*s2) - Q./(2*s2);            % likelihood (prod(w)^(1/2) dropped)
            if strcmp(name, 'marginal_S0noise')
                v   = k.*s2./c;
                lf  = lf - 0.5*log(2*pi*v) - S0.^2./(2*v);  % Zellner g-prior
            else
                lf  = lf - log(2*A);                        % flat, truncated at +/- A
                lf(abs(S0) > A) = -Inf;
            end
            % p(sigma^2) ∝ 1/sigma^2 (-t) times d sigma^2 = sigma^2 dt (+t), and dS0 = sqrt(sigma^2/c) dz
            lf  = lf + 0.5*(t - log(c));
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
