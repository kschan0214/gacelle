classdef McmcBayesUnitTest < matlab.unittest.TestCase
    % Unit tests for the EXPERIMENTAL mcmc_bayes subclass.
    % Phase 0: legacy-option detection (mcmc_bayes.isLegacy).
    % Phase 1: parameter transforms, per-parameter parsing, R-hat/ESS and
    % a small GPU run of the new sampling path.
    % Phase 2: marginal likelihoods vs numerical integration, weighted form,
    % parameter dropping/restoring, S0Param errors, nuisance draws, GPU runs.
    % Phase 3: m counts non-zero weights only, Bartlett inverse-Wishart moments,
    % NIW posterior parameters vs a closed form, Gibbs-block stationary moments
    % (NIW and Jeffreys-half), hierarchical option validation, log-prior cache
    % consistency and GPU runs of the hierarchical prior.
    % Phase 4: MRF neighbour tables and colouring (both modes, several radii,
    % masks with holes and volume edges), edge-weight validation, T4.1 (local
    % vs global MRF energy for every potential), MRF option errors, GPU runs
    % of the chromatic sweep and of run_two_stage.
    %
    % All tests except testNewPath*, testFusedKernel*, testRecoverNuisance*, testHierarchicalCache*,
    % testMrfRuns, testRunTwoStage and testAdaptCovariance{Runs,CacheConsistency,CorrelatedToy,NoSwitch}
    % are pure math and need no GPU.
    % Adaptive covariance (fitting.adaptCovariance): batched Cholesky, Welford moments, refresh fallback,
    % option errors, cache consistency (plain/hierarchical/MRF, full and subsetForward colour steps) and
    % a correlated 2D linear-Gaussian toy with known posterior.
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
    %   Phase 3:
    %   zero weights: logL/stats with zero-weight rows == logL/stats with those rows removed : AbsTol 1e-12 (double)
    %   post-hoc sigma^2 draws with per-voxel m: mean within 5 MC SE of b/(a-1), a = m_v/2 (GPU)
    %   Bartlett IW (host, double), d = 2 and 3, nu = d + 20, N = 1e5 draws: every entry's mean
    %     within 5 SE of Psi/(nu-d-1) and every entry's variance within 5 SE of
    %     [(nu-d+1) psi_ij^2 + (nu-d-1) psi_ii psi_jj] / [(nu-d)(nu-d-1)^2(nu-d-3)]
    %     (SE from the sample: std(x)/sqrt(N), std((x-mean)^2)/sqrt(N)); iwishrnd (if available)
    %     means agree with ours within 5 combined SE
    %   NIW posterior parameters vs direct closed form (Psi_n = Psi0 + sum u u' + k0 m0 m0' - kn mn mn'):
    %     RelTol 1e-10; log NIW_post - log NIW_prior - sum log N(u_i) constant over 6 random (mu,Sigma):
    %     max deviation <= 1e-8 (double)
    %   Gibbs block with fixed u (host, n = 12, d = 2): 'niw' (iid, N = 4e4) and 'jeffreys_half'
    %     (Gibbs chain, N = 1e5, SE from ESS): E[mu], Cov[mu] and E[Sigma] entries within 5 SE of the
    %     analytic marginals (multivariate t: niw Cov = Psi_n/(kn(nun-d-1)); jeffreys Cov = S/(n(n-2d-2)),
    %     E[Sigma] = S/(n-2d-2))
    %   log-prior / log-likelihood / log-Jacobian caches (GPU, checkCache): max |cached - fresh| <= 1e-5
    %     (0 expected: identical deterministic computation)
    %   Phase 4:
    %   neighbour table vs brute-force enumeration of all masked voxel pairs (Chebyshev distance <= r,
    %     Manhattan distance 1 for 'face', same slice for '2d'): exact; symmetry and proper colouring:
    %     exact; colour counts 2 / (r+1)^3 / (r+1)^2 on a 9^3 box: exact
    %   T4.1 local vs global MRF energy (single, CPU and GPU): |dPhi_local - dPhi_bruteforce| <=
    %     1e-5 * S + 1e-6, S = sum of |terms| of the local difference (double brute force over all edges)
    %   MRF GPU runs (checkCache): voxels outside the active colour never move (exactly 0);
    %     caches <= 1e-5 as above
    %   Phase 4b (subsetForward): the setup check is used for lingauss_fwd and ivim_fwd (and records
    %     maxRelDiff <= 1e-6), falls back with mcmc_bayes:subsetForwardFallback for a FWD with a
    %     voxel-dimensioned varargin input (differs / errors / wrong size); the fallback run is
    %     bitwise identical to subsetForward = false (same seed); with the subset, the caches equal a
    %     fresh full-volume recompute within 1e-5 and inactive voxels never move (exactly 0)
    %   Adaptive covariance:
    %   chol_batch vs chol(.,'lower') on random SPD [d,d,50], d = 1..8 (double, CPU and GPU):
    %     max |L - Lref| <= 1e-10 * max|Lref|; indefinite / NaN pages flagged (ok = false)
    %   welford_update vs mean/cov of a stored sample (d = 4, 30 voxels, 500 states, double): RelTol 1e-10
    %   acov_refresh: usable voxel == chol of the regularised covariance (RelTol 1e-10); zero-variance voxel
    %     and voxel with < d+1 acceptances keep their previous factor exactly
    %   caches with adaptCovariance (checkCache): <= 1e-5; MRF inactive voxels never move (exactly 0);
    %     subsetForward true vs false with the elementwise ivim_fwd: identical posterior (same seed)
    %   correlated toy (posterior corr -0.995, 200 voxels, 4000 kept iterations, flat prior, known noise):
    %     per-voxel posterior mean: mean over voxels of z^2 in [0.5, 2], z = (mean - exact)/sqrt(var/ESS);
    %     pooled posterior covariance (samples minus the exact per-voxel mean): every entry within
    %     5 SE of the exact s^2 (A'A)^-1, SE = sqrt(2/sum_voxels ESS) relative;
    %     median ESS/iteration of u1 >= 5 x that of adaptCovariance = false (same seed and burn-in)
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
            'adaptCovariance',          {{'adaptCovariance', true}}, ...
            'overdisp',                 {{'overdisp', 0.01}}, ...
            'prior',                    {{'prior', struct('hierarchical', struct())}}, ...
            'fixedParams',              {{'fixedParams', struct('noise', 0.1)}} ...
            )

        % MRF geometry configurations {mode, radius, connectivity, K}
        mrfGeometry = struct( ...
            'face3d',   {{'3d', 1, 'face', 6}}, ...
            'full3dR1', {{'3d', 1, 'full', 26}}, ...
            'full3dR2', {{'3d', 2, 'full', 124}}, ...
            'face2d',   {{'2d', 1, 'face', 4}}, ...
            'full2dR1', {{'2d', 1, 'full', 8}}, ...
            'full2dR2', {{'2d', 2, 'full', 24}}, ...
            'full2dR3', {{'2d', 3, 'full', 48}} ...
            )

        mrfPotential = {'l1','huber','quadratic'}

        hyperprior = {'niw','jeffreys_half'}

        updateScheme = {'joint','componentwise'}

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

        %% Phase 3: m counts measurements with non-zero weight only
        function testMarginalCountsNonZeroWeights(testCase)
            rng(12);
            m = 12; Nv = 4;
            g = exp(-rand(m,Nv)*3); y = (0.8+0.4*rand(1,Nv)).*g + 0.02*randn(m,Nv);
            w = 0.2 + rand(m,Nv);
            % different zero-weight rows per voxel (voxel 1: none)
            zeroRows = {[], [2 7], [1 5 9 12], 3:6};
            for v = 1:Nv; w(zeroRows{v}, v) = 0; end
            for name = {'marginal_noise','marginal_S0noise','marginal_S0noise_flat'}
                [l, st] = mcmc_bayes.loglik_marginal(g, y, w, name{1}, 0);
                for v = 1:Nv
                    keep = setdiff(1:m, zeroRows{v});
                    [lr, sr] = mcmc_bayes.loglik_marginal(g(keep,v), y(keep,v), w(keep,v), name{1}, 0);
                    testCase.verifyEqual(l(v), lr, 'AbsTol', 1e-12, sprintf('%s voxel %d', name{1}, v));
                    testCase.verifyEqual(st(:,v), sr, 'AbsTol', 1e-12, sprintf('%s stats voxel %d', name{1}, v));
                end
                % explicit m argument gives the same result
                testCase.verifyEqual(mcmc_bayes.loglik_marginal(g, y, w, name{1}, 0, sum(w ~= 0, 1)), l);
            end
            % the legacy m (all rows) would give a different value, so the test is discriminative
            lAll = mcmc_bayes.loglik_marginal(g, y, w, 'marginal_S0noise', 0, m);
            testCase.verifyGreaterThan(abs(lAll(2) - mcmc_bayes.loglik_marginal(g(:,2), y(:,2), w(:,2), 'marginal_S0noise', 0)), 0.1);
        end

        function testRecoverNuisancePerVoxelM(testCase)
            gacelletest.assumeGPU(testCase);
            fitting.modelParams = {'R2star';'noise'};
            fitting.lb = [0.1; 0.001]; fitting.ub = [200; 0.1]; fitting.xStepSize = [1; 0.005];
            fitting.likelihood = 'marginal_noise';
            [~, lik] = mcmc_bayes.setup_likelihood(fitting);
            mV = [12 7]; R = 0.004; Nv = 2; Ns = 1e5; Nrep = 2;
            statsPost = repmat(single(R), 1, Nv, Ns, Nrep);
            xPost.R2star = rand(Nv, Ns, Nrep, 'single');
            parallel.gpu.rng(6);
            xOut = mcmc_bayes.recover_nuisance(xPost, statsPost, lik, mV);
            for v = 1:Nv
                a   = mV(v)/2; b = R/2; N = Ns*Nrep;
                s2  = double(reshape(xOut.noise(v,:,:), [], 1)).^2;
                Es2 = b/(a-1); Vs2 = b^2/((a-1)^2*(a-2));
                testCase.verifyLessThanOrEqual(abs(mean(s2) - Es2), 5*sqrt(Vs2/N), sprintf('voxel %d', v));
            end
        end

        function testNewPathZeroWeightVoxelErrors(testCase)
            gacelletest.assumeGPU(testCase);
            [y, mask, w, pars0, fitting, obj] = McmcBayesUnitTest.r2starSetup();
            f = fitting; f.likelihood = 'marginal_noise';
            w = ones(size(y), 'single'); w(1,1,1,:) = 0;                % one voxel without data
            testCase.verifyError(@() mcmc_bayes().optimisation(y, mask, w, pars0, f, @obj.FWD, 'mcmc', f), ...
                'mcmc_bayes:tooFewMeasurements');
        end

        %% Phase 3: inverse-Wishart (Bartlett) and NIW posterior (host, double)
        function testDrawIWBartlettMoments(testCase)
            for d = [2 3]
                rng(300 + d);
                B   = randn(d); Psi = B*B.' + d*eye(d);
                nu  = d + 20; N = 1e5;
                X   = zeros(d, d, N);
                for k = 1:N; X(:,:,k) = mcmc_bayes.draw_iw_bartlett(Psi, nu); end
                testCase.verifyEqual(X, permute(X, [2 1 3]), 'draws must be symmetric');
                E   = Psi ./ (nu - d - 1);
                V   = ((nu-d+1).*Psi.^2 + (nu-d-1).*diag(Psi)*diag(Psi).') ./ ((nu-d)*(nu-d-1)^2*(nu-d-3));
                Xf  = reshape(X, d*d, N);
                mS  = mean(Xf, 2); sdS = std(Xf, 0, 2);
                dev = (Xf - mS).^2;
                vS  = mean(dev, 2) * N/(N-1); seV = std(dev, 0, 2) ./ sqrt(N);
                zM  = (mS - E(:)) ./ (sdS ./ sqrt(N));
                zV  = (vS - V(:)) ./ seV;
                testCase.verifyLessThanOrEqual(max(abs(zM)), 5, sprintf('d=%d mean z = %s', d, mat2str(zM.', 3)));
                testCase.verifyLessThanOrEqual(max(abs(zV)), 5, sprintf('d=%d var z = %s', d, mat2str(zV.', 3)));

                % optional cross-check against the Statistics toolbox (not a dependency)
                if exist('iwishrnd', 'file') == 2 && license('test', 'Statistics_Toolbox')
                    Y   = zeros(d*d, N);
                    for k = 1:N; Yk = iwishrnd(Psi, nu); Y(:,k) = Yk(:); end
                    zI  = (mS - mean(Y,2)) ./ sqrt(sdS.^2./N + var(Y,0,2)./N);
                    testCase.verifyLessThanOrEqual(max(abs(zI)), 5, sprintf('d=%d ours vs iwishrnd z = %s', d, mat2str(zI.', 3)));
                end
            end
        end

        function testNiwPosteriorClosedForm(testCase)
            rng(310);
            d = 3; n = 7;
            u = randn(d, n) .* [1; 2; 0.5] + [0.3; -1; 2];
            m0 = [0.1; 0.2; -0.3]; kappa0 = 0.7; nu0 = d + 3;
            B = randn(d); Psi0 = B*B.' + eye(d);
            [ubar, S] = mcmc_bayes.hyper_suffstats(u);
            [mn, kn, Psin, nun] = mcmc_bayes.niw_posterior(ubar, S, n, m0, kappa0, Psi0, nu0);

            % direct closed form from the raw sums
            knD   = kappa0 + n; nunD = nu0 + n;
            mnD   = (kappa0*m0 + sum(u,2)) / knD;
            PsinD = Psi0 + u*u.' + kappa0*(m0*m0.') - knD*(mnD*mnD.');
            testCase.verifyEqual([kn nun], [knD nunD]);
            testCase.verifyEqual(mn, mnD, 'RelTol', 1e-10);
            testCase.verifyEqual(Psin, (PsinD+PsinD.')/2, 'RelTol', 1e-10);

            % Bayes' rule: log NIW_post(mu,Sigma) - log NIW_prior(mu,Sigma) - sum_i log N(u_i|mu,Sigma)
            % must not depend on (mu, Sigma)
            c = zeros(1, 6);
            for k = 1:6
                Bk = randn(d); Sigma = Bk*Bk.' + 0.5*eye(d); mu = randn(d,1);
                ll = 0;
                for i = 1:n; r = u(:,i) - mu; ll = ll - 0.5*log(det(Sigma)) - 0.5*(r.'/Sigma)*r; end
                c(k) = McmcBayesUnitTest.logNIW(mu, Sigma, mn, kn, Psin, nun) - ...
                       McmcBayesUnitTest.logNIW(mu, Sigma, m0, kappa0, Psi0, nu0) - ll;
            end
            testCase.verifyLessThanOrEqual(max(abs(c - c(1))), 1e-8);
        end

        function testGibbsBlockStationaryMoments(testCase, hyperprior)
            rng(320);
            d = 2; n = 12;
            u = randn(d, n) .* [1; 0.5] + [1; -2];
            [ubar, S] = mcmc_bayes.hyper_suffstats(u);
            hp = struct('hyperprior', hyperprior, 'm0', [0; 0], 'kappa0', 0.5, 'Psi0', eye(d), 'nu0', d + 2);
            switch hyperprior
                case 'niw'
                    N = 4e4;
                    [mn, kn, Psin, nun] = mcmc_bayes.niw_posterior(ubar, S, n, hp.m0, hp.kappa0, hp.Psi0, hp.nu0);
                    Emu = mn; Cmu = Psin/(kn*(nun-d-1)); ESig = Psin/(nun-d-1);
                case 'jeffreys_half'
                    N = 1e5;
                    Emu = ubar; Cmu = S/(n*(n-2*d-2)); ESig = S/(n-2*d-2);
            end
            mu = ubar; Sigma = S/n;
            MU = zeros(d, N); SIG = zeros(d*d, N);
            for k = 1:N
                [mu, Sigma] = mcmc_bayes.gibbs_hyper(ubar, S, n, mu, Sigma, hp);
                MU(:,k) = mu; SIG(:,k) = Sigma(:);
            end
            % statistics with their target: E[mu_p], E[(mu_p-Emu_p)(mu_q-Emu_q)], E[Sigma_pq]
            dm  = MU - Emu;
            T   = [MU; dm(1,:).^2; dm(1,:).*dm(2,:); dm(2,:).^2; SIG];
            ref = [Emu; Cmu(1,1); Cmu(1,2); Cmu(2,2); ESig(:)];
            ess = mcmc_bayes.ess(reshape(T, size(T,1), N, 1));
            z   = (mean(T,2) - ref) ./ (std(T,0,2) ./ sqrt(ess));
            testCase.verifyLessThanOrEqual(max(abs(z)), 5, sprintf('%s: z = %s', hyperprior, mat2str(z.', 3)));
        end

        %% Phase 3: hierarchical options (pure, no GPU)
        function testHierarchicalSetupDefaultsAndErrors(testCase)
            f.modelParams = {'M0';'R2star';'noise'};
            f.lb = [0; 0; 0.001]; f.ub = [2; Inf; 0.1]; f.xStepSize = [0.01; 1; 0.005];
            f.parameterTransform = {'sigmoid','log','linear'};
            f.prior.hierarchical = struct();
            h = mcmc_bayes.setup_hierarchical(f);
            testCase.verifyTrue(h.on);
            testCase.verifyEqual(h.params, {'M0','R2star'});        % default: all sampled but noise
            testCase.verifyEqual(h.idx, [1 2]);
            testCase.verifyEqual(h.hyperprior, 'niw');
            testCase.verifyEqual(h.kappa0, 1e-3);
            testCase.verifyFalse(h.fixed);
            f.prior.hierarchical = true;
            testCase.verifyTrue(mcmc_bayes.setup_hierarchical(f).on);
            f.prior.hierarchical = false;
            testCase.verifyFalse(mcmc_bayes.setup_hierarchical(f).on);
            f.prior = struct();
            testCase.verifyFalse(mcmc_bayes.setup_hierarchical(f).on);

            % explicit params, model order kept
            f.prior.hierarchical = struct('params', {{'R2star'}});
            h = mcmc_bayes.setup_hierarchical(f);
            testCase.verifyEqual(h.params, {'R2star'}); testCase.verifyEqual(h.d, 1);

            % transform/bound combinations that would need bound rejection
            bad = { {'sigmoid','log','linear'}, [0; 0.1; 0.001], [2; Inf; 0.1];     % log with lb > 0
                    {'sigmoid','log','linear'}, [0; 0; 0.001],   [2; 200; 0.1];     % log with finite ub
                    {'linear','log','linear'},  [0; 0; 0.001],   [2; Inf; 0.1];     % linear with finite bounds
                    {'sigmoid','log','linear'}, [0; 0; 0.001],   [Inf; Inf; 0.1] }; % sigmoid with infinite ub
            for k = 1:size(bad,1)
                g = f; g.prior.hierarchical = struct();
                g.parameterTransform = bad{k,1}; g.lb = bad{k,2}; g.ub = bad{k,3};
                testCase.verifyError(@() mcmc_bayes.setup_hierarchical(g), 'mcmc_bayes:hierarchicalTransform', sprintf('case %d', k));
            end
            % 'noise' included explicitly: linear with a finite box is not allowed either
            g = f; g.prior.hierarchical = struct('params', {{'M0','noise'}});
            testCase.verifyError(@() mcmc_bayes.setup_hierarchical(g), 'mcmc_bayes:hierarchicalTransform');

            % other errors
            g = f; g.prior.hierarchical = struct('params', {{'S0'}});
            testCase.verifyError(@() mcmc_bayes.setup_hierarchical(g), 'mcmc_bayes:hierarchicalParams');
            g = f; g.prior.hierarchical = struct('hyperprior', 'wishart');
            testCase.verifyError(@() mcmc_bayes.setup_hierarchical(g), 'mcmc_bayes:invalidPrior');
            g = f; g.prior.hierarchical = struct('tau', 1);
            testCase.verifyError(@() mcmc_bayes.setup_hierarchical(g), 'mcmc_bayes:invalidPrior');
            g = f; g.prior.hierarchical = struct('Psi0', [1 2; 2 1]);                   % not positive definite
            testCase.verifyError(@() mcmc_bayes.setup_hierarchical(g), 'mcmc_bayes:invalidPrior');
            g = f; g.prior.hierarchical = struct('nu0', 0.5);                           % nu0 <= d-1
            testCase.verifyError(@() mcmc_bayes.setup_hierarchical(g), 'mcmc_bayes:invalidPrior');
            g = f; g.prior.hierarchical = struct('fixed', true);                        % no mu/Sigma
            testCase.verifyError(@() mcmc_bayes.setup_hierarchical(g), 'mcmc_bayes:hierarchicalFixed');
            g = f; g.prior.hierarchical = struct('fixed', true, 'mu', [0 0], 'Sigma', [1 0; 0 -1]);
            testCase.verifyError(@() mcmc_bayes.setup_hierarchical(g), 'mcmc_bayes:hierarchicalFixed');
            g = f; g.prior.hierarchical = struct('mu', [0 0]);                          % mu without fixed
            testCase.verifyError(@() mcmc_bayes.setup_hierarchical(g), 'mcmc_bayes:hierarchicalFixed');
            g = f; g.prior.hierarchical = struct('subsetFraction', 1.5);
            testCase.verifyError(@() mcmc_bayes.setup_hierarchical(g), 'mcmc_bayes:subsetFraction');
            g = f; g.prior = struct('hierarchical', struct(), 'spatial', 1);
            testCase.verifyError(@() mcmc_bayes.setup_hierarchical(g), 'mcmc_bayes:invalidPrior');
            g = f; g.prior.hierarchical = struct('fixed', true, 'mu', [0 3], 'Sigma', [1 0.2; 0.2 2]);
            h = mcmc_bayes.setup_hierarchical(g);
            testCase.verifyEqual(h.mu, [0; 3]);

            % through optimisation (errors raised before any GPU code)
            run = @(ff) mcmc_bayes().optimisation([], [], [], [], ff, []);
            g = f; g.prior.hierarchical = struct('subsetFraction', 0.5);
            testCase.verifyError(@() run(g), 'mcmc_bayes:subsetFraction');
            g = f; g.lb(2) = 0.1; g.prior.hierarchical = struct();
            testCase.verifyError(@() run(g), 'mcmc_bayes:hierarchicalTransform');
            % marginalised parameters cannot be under the hierarchy
            g = f; g.likelihood = 'marginal_S0noise'; g.S0Param = 'M0';
            g.prior.hierarchical = struct('params', {{'M0','R2star'}});
            testCase.verifyError(@() run(g), 'mcmc_bayes:hierarchicalParams');
            g.prior.hierarchical = struct();                                            % default: R2star only
            [gs, ~] = mcmc_bayes.setup_likelihood(mcmc_bayes.check_set_default_bayes(g));
            testCase.verifyEqual(mcmc_bayes.setup_hierarchical(gs).params, {'R2star'});
        end

        function testFixedParamsTestOnly(testCase)
            f.modelParams = {'u1';'u2';'noise'}; f.lb = [-Inf; -Inf; 0]; f.ub = [Inf; Inf; 1]; f.xStepSize = [0.1; 0.1; 0.01];
            f.fixedParams = struct('noise', 0.05);
            testCase.verifyFalse(mcmc_bayes.isLegacy(f));
            [fs, lik] = mcmc_bayes.setup_likelihood(f);
            testCase.verifyEqual(fs.modelParams, {'u1';'u2'});
            testCase.verifyEqual(lik.fittingOut.modelParams, {'u1';'u2'});
            testCase.verifyEqual(lik.userFixed.noise, 0.05);
            g = f; g.fixedParams = struct('sigma', 1);
            testCase.verifyError(@() mcmc_bayes.setup_likelihood(g), 'mcmc_bayes:fixedParams');
            g = f; g.fixedParams = struct('noise', [1 2]);
            testCase.verifyError(@() mcmc_bayes.setup_likelihood(g), 'mcmc_bayes:fixedParams');
            g = f; g.likelihood = 'marginal_noise';                     % noise is marginalised, cannot be fixed
            testCase.verifyError(@() mcmc_bayes.setup_likelihood(g), 'mcmc_bayes:fixedParams');
        end

        %% Phase 3: GPU runs of the hierarchical prior
        function testHierarchicalCacheConsistency(testCase, hyperprior, updateScheme)
            gacelletest.assumeGPU(testCase);
            [yy, mask, x0, f, fwd] = McmcBayesUnitTest.linGaussSetup(40, 2);
            f.prior.hierarchical = struct('hyperprior', hyperprior);
            f.updateScheme  = updateScheme;
            f.checkCache    = true;
            f.repetition    = 2; f.overdisp = 0.01;
            out = mcmc_bayes().optimisation(yy, mask, [], x0, f, fwd);
            cc  = out.diagnostics.cacheCheck;
            testCase.verifyEqual(cc.Ncheck, f.iteration*f.repetition);
            testCase.verifyLessThanOrEqual(cc.logprior, 1e-5, sprintf('logprior cache error %g', cc.logprior));
            testCase.verifyLessThanOrEqual(cc.loglik,   1e-5, sprintf('loglik cache error %g', cc.loglik));
            testCase.verifyLessThanOrEqual(cc.logjac,   1e-5, sprintf('logjac cache error %g', cc.logjac));

            % mixed transforms under the marginal likelihood (Jacobian dropped for the hierarchical rows only)
            [y, mask, w, pars0, fitting, obj] = McmcBayesUnitTest.r2starSetup();
            g = fitting;
            g.likelihood = 'marginal_noise'; g.updateScheme = updateScheme; g.checkCache = true;
            g.lb = [0; 0; 0.001]; g.ub = [2; Inf; 0.1];
            g.parameterTransform = {'sigmoid','log','linear'};
            g.prior.hierarchical = struct('hyperprior', hyperprior, 'params', {{'R2star'}});   % M0 keeps its Jacobian
            out = mcmc_bayes().optimisation(y, mask, w, pars0, g, @obj.FWD, 'mcmc', g);
            cc  = out.diagnostics.cacheCheck;
            testCase.verifyLessThanOrEqual(max([cc.logprior cc.loglik cc.logjac]), 1e-5);
        end

        function testNewPathHierarchicalRuns(testCase)
            gacelletest.assumeGPU(testCase);
            [y, mask, w, pars0, fitting, obj] = McmcBayesUnitTest.r2starSetup();
            f = fitting;
            f.lb = [0; 0; 0.001]; f.ub = [2; Inf; 0.1];
            f.parameterTransform = {'sigmoid','log','linear'};
            f.prior.hierarchical = struct();                            % M0, R2star; noise keeps its box
            f.repetition = 2; f.overdisp = 0.01; f.adaptStepSize = true; f.adaptInterval = 10;
            out = mcmc_bayes().optimisation(y, mask, w, pars0, f, @obj.FWD, 'mcmc', f);
            Ns  = numel(21:2:200);
            testCase.verifyEqual(out.hyper.params, {'M0','R2star'});
            testCase.verifyEqual(size(out.hyper.posterior.mu), [2 Ns 2]);
            testCase.verifyEqual(size(out.hyper.posterior.Sigma), [2 2 Ns 2]);
            testCase.verifyEqual(size(out.hyper.rhat.Sigma), [2 2]);
            testCase.verifyEqual(size(out.hyper.ess.mu), [2 1]);
            testCase.verifyEqual(out.hyper.transform{2}, 'u = log(x)');
            testCase.verifyTrue(all(isfinite(out.hyper.mean.Sigma(:))));
            testCase.verifyEqual(out.settings.prior.hierarchical.nu0, 4);
            testCase.verifyEqual(out.settings.prior.hierarchical.kappa0, 1e-3);
            testCase.verifyEqual(fieldnames(out.posterior), {'M0';'R2star';'noise'});
            testCase.verifyGreaterThanOrEqual(min(out.posterior.noise(:)), single(f.lb(3)));
            testCase.verifyLessThanOrEqual(max(out.posterior.noise(:)), single(f.ub(3)));
            testCase.verifyGreaterThan(min(out.posterior.R2star(:)), 0);
            testCase.verifyTrue(all(isfinite(out.mean.R2star(:))));

            % memory guard
            g = f; g.prior.hierarchical = struct('maxGPUMemory', 1);
            testCase.verifyError(@() mcmc_bayes().optimisation(y, mask, w, pars0, g, @obj.FWD, 'mcmc', g), ...
                'mcmc_bayes:hierarchicalMemory');
            % fixed mode needs no memory guard
            g.prior.hierarchical = struct('maxGPUMemory', 1, 'fixed', true, 'mu', [0 3.4], 'Sigma', eye(2));
            out = mcmc_bayes().optimisation(y, mask, w, pars0, g, @obj.FWD, 'mcmc', g);
            testCase.verifyTrue(out.hyper.fixed);
            testCase.verifyEqual(out.hyper.mean.mu, [0; 3.4]);

            % stage 1 on a voxel subset, then fixed mode
            g = f; g.repetition = 1; g.prior.hierarchical = struct('subsetFraction', 0.5);
            rng(3);
            [muHat, SigmaHat, fFixed, outSub, idx] = mcmc_bayes().estimate_hyper_subset(y, mask, w, pars0, g, @obj.FWD, 'mcmc', g);
            testCase.verifyEqual(numel(idx), 8);                        % min(Nmask, max(ceil(0.5*8), 10))
            testCase.verifyEqual(muHat, outSub.hyper.mean.mu);
            testCase.verifyEqual(SigmaHat, outSub.hyper.mean.Sigma);
            testCase.verifyTrue(fFixed.prior.hierarchical.fixed);
            out = mcmc_bayes().optimisation(y, mask, w, pars0, fFixed, @obj.FWD, 'mcmc', fFixed);
            testCase.verifyEqual(out.settings.prior.hierarchical.Sigma, SigmaHat);
        end

        %% Phase 4: MRF geometry (pure, no GPU)
        function testMrfNeighboursAndColouring(testCase, mrfGeometry)
            [mode, r, conn, K] = mrfGeometry{:};
            % masks: full box (volume edges) and a box with random holes (mask edges)
            rng(400);
            masks = {true(7,6,5), rand(8,7,5) > 0.3, true(9,1,1), true(5,6)};
            for km = 1:numel(masks)
                mask     = masks{km};
                mask_idx = find(mask);
                dims     = size(mask);
                nbr      = mcmc_bayes.build_neighbours(mask_idx, dims, mode, r, conn);
                testCase.verifyClass(nbr, 'int32');
                testCase.verifySize(nbr, [K numel(mask_idx)]);
                % symmetry (also asserted inside), checked independently here
                [kk, v] = find(nbr > 0);
                n = double(nbr(sub2ind(size(nbr), kk, v)));
                testCase.verifyEqual(double(nbr(sub2ind(size(nbr), K+1-kk, n))), v, sprintf('mask %d symmetry', km));
                % neighbour sets vs brute-force enumeration of all masked pairs
                d3 = [dims ones(1, 3-numel(dims))];
                [i, j, l] = ind2sub(d3, mask_idx);
                P  = [i j l];
                D  = abs(permute(P, [1 3 2]) - permute(P, [3 1 2]));        % [Nv, Nv, 3]
                if strcmp(conn, 'face'); A = sum(D, 3) == 1; else; A = max(D, [], 3) <= r & max(D, [], 3) > 0; end
                if strcmp(mode, '2d'); A = A & D(:,:,3) == 0; end
                B  = false(numel(mask_idx));
                B(sub2ind(size(B), v, n)) = true;
                testCase.verifyEqual(B, A, sprintf('mask %d neighbour sets', km));
                % proper colouring, colours in 1..Nc
                [col, Nc] = mcmc_bayes.build_colours(mask_idx, dims, mode, r, conn, nbr);
                testCase.verifySize(col, [1 numel(mask_idx)]);
                testCase.verifyTrue(all(col >= 1 & col <= Nc));
                testCase.verifyFalse(any(col(v(:).') == col(n(:).')), sprintf('mask %d colouring', km));
            end
            % colour counts: 2 ('face'), (r+1)^3 ('3d') or (r+1)^2 ('2d'); all used on a large box
            mask = true(9,9,9);
            [col, Nc] = mcmc_bayes.build_colours(find(mask), size(mask), mode, r, conn);
            if strcmp(conn, 'face'); NcRef = 2; elseif strcmp(mode, '3d'); NcRef = (r+1)^3; else; NcRef = (r+1)^2; end
            testCase.verifyEqual(Nc, NcRef);
            testCase.verifyEqual(numel(unique(col)), NcRef);
            % an interior voxel has all K neighbours
            nbr = mcmc_bayes.build_neighbours(find(mask), size(mask), mode, r, conn);
            testCase.verifyEqual(nnz(nbr(:, sub2ind(size(mask), 5, 5, 5))), K);
            % an improper colouring is detected
            testCase.verifyError(@() mcmc_bayes.check_colouring(nbr, ones(1, numel(mask))), 'mcmc_bayes:mrfColouring');
        end

        function testMrfNeighbourAsymmetryDetected(testCase)
            mask = true(4,4,2);
            nbr  = mcmc_bayes.build_neighbours(find(mask), size(mask), '3d', 1, 'face');
            k    = find(nbr(:,1) > 0, 1);
            nbr(k,1) = 0;                                   % break one direction of an edge
            testCase.verifyError(@() mcmc_bayes.check_neighbour_symmetry(nbr), 'mcmc_bayes:mrfNeighbours');
        end

        function testMrfEdgeWeights(testCase)
            rng(401);
            mask = rand(6,5,3) > 0.25;
            nbr  = mcmc_bayes.build_neighbours(find(mask), size(mask), '3d', 1, 'full');
            w    = McmcBayesUnitTest.symmetricEdgeWeights(nbr);
            w(nbr == 0) = NaN;                              % entries of absent neighbours are ignored
            wc   = mcmc_bayes.check_edge_weights(w, nbr);
            testCase.verifyEqual(wc(nbr == 0), zeros(nnz(nbr == 0), 1));
            testCase.verifyEqual(wc(nbr > 0), w(nbr > 0));
            % wrong size, negative, non-finite, asymmetric
            testCase.verifyError(@() mcmc_bayes.check_edge_weights(w(1:end-1,:), nbr), 'mcmc_bayes:invalidEdgeWeights');
            [k, v] = find(nbr > 0, 1);
            w2 = w; w2(k,v) = -1;
            testCase.verifyError(@() mcmc_bayes.check_edge_weights(w2, nbr), 'mcmc_bayes:invalidEdgeWeights');
            w2 = w; w2(k,v) = Inf;
            testCase.verifyError(@() mcmc_bayes.check_edge_weights(w2, nbr), 'mcmc_bayes:invalidEdgeWeights');
            w2 = w; w2(k,v) = w2(k,v) + 0.1;
            testCase.verifyError(@() mcmc_bayes.check_edge_weights(w2, nbr), 'mcmc_bayes:invalidEdgeWeights');
        end

        %% Phase 4: T4.1 local vs global MRF energy (CPU single; GPU if available)
        function testMrfLocalMatchesGlobal(testCase, mrfPotential)
            % Tolerance (stated before running): for every perturbation,
            %   |dPhi_local - dPhi_bruteforce| <= 1e-5 * S + 1e-6,
            % with dPhi_local from mcmc_bayes.mrf_local_delta in single precision,
            % dPhi_bruteforce = Phi(u_new) - Phi(u_old) over all edges (each once) in double, and
            % S the sum of |terms| entering the local difference (scale of the single-precision sums).
            geoms = {{'3d',1,'face'}, {'3d',1,'full'}, {'2d',2,'full'}, {'2d',1,'face'}, {'3d',2,'full'}};
            useGPU = canUseGPU();
            for kg = 1:numel(geoms)
                [mode, r, conn] = geoms{kg}{:};
                rng(410 + kg);
                mask = rand(7,6,4) > 0.3;
                mask_idx = find(mask); Nv = numel(mask_idx);
                nbr  = mcmc_bayes.build_neighbours(mask_idx, size(mask), mode, r, conn);
                col  = mcmc_bayes.build_colours(mask_idx, size(mask), mode, r, conn, nbr);
                w    = McmcBayesUnitTest.symmetricEdgeWeights(nbr);
                K    = size(nbr, 1);
                self = repmat(int32(1:Nv), K, 1); nbrS = nbr; nbrS(nbr == 0) = self(nbr == 0);
                d    = 2; W = [0.7; 1.3]; tau = 0.8; delta = [0.3; 0.6];
                u    = randn(d, Nv);
                phi  = @(uu) McmcBayesUnitTest.bruteForcePhi(uu, nbr, w, W, tau, mrfPotential, delta);
                % (a) single-voxel perturbations, all parameters (joint) and one parameter (componentwise)
                for t = 1:15
                    i = randi(Nv);
                    uN = u; uN(:,i) = uN(:,i) + 0.7*randn(d,1);
                    if t == 1; uN(:,i) = u(:,i) + delta; end            % exactly at the Huber threshold
                    ref = phi(uN) - phi(u);
                    loc = mcmc_bayes.mrf_local_delta(single(uN(:,i)), single(u(:,i)), single(u), nbrS(:,i), single(w(:,i)), ...
                                                     single(W/tau), single(delta), mrfPotential);
                    S   = McmcBayesUnitTest.localScale(uN(:,i), u(:,i), u, nbrS(:,i), w(:,i), W/tau, delta, mrfPotential);
                    testCase.verifyLessThanOrEqual(abs(double(loc) - ref), 1e-5*S + 1e-6, sprintf('%s %s r%d joint', mrfPotential, mode, r));
                    p   = randi(d);
                    uN  = u; uN(p,i) = uN(p,i) + 0.7*randn;
                    ref = phi(uN) - phi(u);
                    loc = mcmc_bayes.mrf_local_delta(single(uN(p,i)), single(u(p,i)), single(u(p,:)), nbrS(:,i), single(w(:,i)), ...
                                                     single(W(p)/tau), single(delta(p)), mrfPotential);
                    testCase.verifyLessThanOrEqual(abs(double(loc) - ref), 1e-5*S + 1e-6, sprintf('%s %s r%d componentwise', mrfPotential, mode, r));
                end
                % (b) a whole colour class moved at once: dPhi = sum of the local differences
                for c = unique(col)
                    act = find(col == c);
                    uN  = u; uN(:,act) = uN(:,act) + 0.5*randn(d, numel(act));
                    ref = phi(uN) - phi(u);
                    loc = mcmc_bayes.mrf_local_delta(single(uN(:,act)), single(u(:,act)), single(u), nbrS(:,act), single(w(:,act)), ...
                                                     single(W/tau), single(delta), mrfPotential);
                    S   = McmcBayesUnitTest.localScale(uN(:,act), u(:,act), u, nbrS(:,act), w(:,act), W/tau, delta, mrfPotential);
                    testCase.verifyLessThanOrEqual(abs(sum(double(loc)) - ref), 1e-5*S + 1e-6, sprintf('%s %s r%d colour %d', mrfPotential, mode, r, c));
                    if useGPU
                        locG = mcmc_bayes.mrf_local_delta(gpuArray(single(uN(:,act))), gpuArray(single(u(:,act))), gpuArray(single(u)), ...
                                    gpuArray(uint32(nbrS(:,act))), gpuArray(single(w(:,act))), gpuArray(single(W/tau)), gpuArray(single(delta)), mrfPotential);
                        testCase.verifyLessThanOrEqual(abs(sum(double(gather(locG))) - ref), 1e-5*S + 1e-6, 'GPU');
                    end
                end
            end
        end

        %% Phase 4: MRF option errors (pure, no GPU)
        function testMrfOptionErrors(testCase)
            f.modelParams = {'u1';'u2';'noise'}; f.lb = [-Inf; -Inf; 0]; f.ub = [Inf; Inf; 10]; f.xStepSize = [0.2; 0.2; 0.01];
            f.algorithm = 'MH'; f.fixedParams = struct('noise', 1);
            fixedH = struct('fixed', true, 'mu', [0 0], 'Sigma', [4 0; 0 0.25]);
            run = @(ff) mcmc_bayes().optimisation([], [], [], [], ff, []);
            % mrf without hierarchical
            g = f; g.prior = struct('mrf', struct());
            testCase.verifyError(@() run(g), 'mcmc_bayes:mrfRequiresHierarchical');
            % mrf with free hyperparameters: optimisation dispatches to run_two_stage (Phase 6a);
            % setup_mrf itself still rejects the combination
            g = f; g.prior = struct('hierarchical', struct(), 'mrf', struct('tau', 1));
            testCase.verifyTrue(mcmc_bayes.is_two_stage(g));
            gs = mcmc_bayes.setup_likelihood(mcmc_bayes.check_set_default_bayes(g));
            testCase.verifyError(@() mcmc_bayes.setup_mrf(gs, mcmc_bayes.setup_hierarchical(gs)), 'mcmc_bayes:mrfFreeHyperparameters');
            g.prior.mrf = true;
            testCase.verifyTrue(mcmc_bayes.is_two_stage(g));
            % invalid MRF options
            bad = {struct('potential','tv'), struct('tau',0), struct('tau',[1 2]), struct('W',[1 -1]), struct('W',[1 1 1]), ...
                   struct('mode','1d'), struct('radius',1.5), struct('radius',2,'connectivity','face'), ...
                   struct('connectivity','edge'), struct('huberDelta',0), struct('beta',1), ...
                   struct('subsetForward','yes'), struct('subsetForward',[true true]), struct('subsetForward',2), ...
                   struct('bayesivimWeights','yes'), struct('bayesivimWeights',true,'potential','huber'), ...
                   struct('bayesivimWeights',true,'W',[1 1])};
            for k = 1:numel(bad)
                g = f; g.prior = struct('hierarchical', fixedH, 'mrf', bad{k});
                testCase.verifyError(@() run(g), 'mcmc_bayes:invalidMrf', sprintf('case %d', k));
            end
            % test-only simultaneous update needs an MRF; unknown update name
            g = f; g.prior = struct('hierarchical', fixedH); g.mrfUpdate = 'simultaneous';
            testCase.verifyError(@() run(g), 'mcmc_bayes:invalidMrf');
            g.prior.mrf = struct(); g.mrfUpdate = 'jacobi';
            testCase.verifyError(@() run(g), 'mcmc_bayes:invalidMrf');
            % run_two_stage needs a free hierarchical prior and an MRF
            g = f; g.prior = struct('hierarchical', fixedH, 'mrf', struct());
            testCase.verifyError(@() mcmc_bayes().run_two_stage([], [], [], [], g, []), 'mcmc_bayes:invalidPrior');
            g = f; g.prior = struct('hierarchical', struct());
            testCase.verifyError(@() mcmc_bayes().run_two_stage([], [], [], [], g, []), 'mcmc_bayes:invalidPrior');

            % resolved defaults: W = 1/sqrt(diag(Sigma)), delta = huberDelta*sqrt(diag(Sigma)), connectivity
            g = f; g.prior = struct('hierarchical', fixedH, 'mrf', struct());
            fs = mcmc_bayes.setup_likelihood(mcmc_bayes.check_set_default_bayes(g));
            m  = mcmc_bayes.setup_mrf(mcmc_bayes.check_set_default_bayes(fs), mcmc_bayes.setup_hierarchical(fs));
            testCase.verifyEqual(m.W, [0.5; 2]);
            testCase.verifyEqual(m.delta, [2; 0.5]);
            testCase.verifyEqual({m.potential, m.mode, m.connectivity, m.radius, m.tau, m.update}, {'l1','3d','face',1,1,'chromatic'});
            testCase.verifyTrue(m.subsetForward);                  % Phase 4b default
            testCase.verifyFalse(m.stateWeight);                   % TEST ONLY bayesivimWeights off by default
            g2 = g; g2.prior.mrf = struct('bayesivimWeights', true);
            m2 = mcmc_bayes.setup_mrf(mcmc_bayes.check_set_default_bayes(g2), mcmc_bayes.setup_hierarchical(mcmc_bayes.setup_likelihood(g2)));
            testCase.verifyTrue(m2.stateWeight);
            testCase.verifyEqual(m2.W, [1; 1]);
            g.prior.mrf = struct('mode','2d','radius',2,'huberDelta',0.5);
            m  = mcmc_bayes.setup_mrf(g, mcmc_bayes.setup_hierarchical(mcmc_bayes.setup_likelihood(g)));
            testCase.verifyEqual(m.connectivity, 'full');
            testCase.verifyEqual(m.delta, [1; 0.25]);
        end

        %% TEST ONLY bayesivimWeights: local term vs BayesIVIM's formula written out directly
        function testBayesivimWeightsLocalDelta(testCase)
            % Tolerance (stated before running): |dPhi_local - dPhi_direct| <= 1e-5 * max(1, |dPhi_direct|),
            % single precision vs double. BayesIVIM (ivim_bayes.m, wSumNeighb): for voxel i and parameter p,
            % S(u) = sum over the (2r+1)^2 in-mask block INCLUDING the centre of |u_block,curr - u|,
            % and the spatial ratio is exp(-(1/tau) (S(u_prop) - S(u_curr)) / |u_i,curr|).
            rng(812);
            mask = rand(9,8) > 0.25; mask_idx = find(mask); Nv = numel(mask_idx);
            nbr  = mcmc_bayes.build_neighbours(mask_idx, [size(mask) 1], '2d', 2, 'full');
            K    = size(nbr,1); self = repmat(int32(1:Nv), K, 1); nbrS = nbr; nbrS(nbr == 0) = self(nbr == 0);
            w    = double(nbr > 0);
            tau  = 0.7;
            u    = -3 + randn(1, Nv);                          % one parameter, away from 0 (weight 1/|u|)
            img  = nan(size(mask)); img(mask_idx) = u;
            for t = 1:20
                i  = randi(Nv); uNew = u(i) + 0.5*randn;
                [ix, iy] = ind2sub(size(mask), mask_idx(i));
                blk = img(max(1,ix-2):min(end,ix+2), max(1,iy-2):min(end,iy+2)); blk = blk(~isnan(blk));   % includes centre
                Sn  = sum(abs(blk - uNew)); So = sum(abs(blk - u(i)));
                ref = (Sn - So) / abs(u(i)) / tau;
                loc = mcmc_bayes.mrf_local_delta(single(uNew), single(u(i)), single(u), nbrS(:,i), single(w(:,i)), ...
                                                 single(1/tau), single(1), 'l1', true);
                testCase.verifyLessThanOrEqual(abs(double(loc) - ref), 1e-5*max(1, abs(ref)), sprintf('trial %d', t));
            end
            % without the flag the same call is the ordinary L1 term (no centre, no 1/|u|)
            i = 1; uNew = u(i) + 0.3;
            loc0 = mcmc_bayes.mrf_local_delta(single(uNew), single(u(i)), single(u), nbrS(:,i), single(w(:,i)), single(1/tau), single(1), 'l1');
            ref0 = sum(w(:,i) .* (abs(uNew - u(nbrS(:,i))') - abs(u(i) - u(nbrS(:,i))'))) / tau;
            testCase.verifyLessThanOrEqual(abs(double(loc0) - ref0), 1e-5*max(1, abs(ref0)));
        end

        %% Phase 6a: helpers used by the model wrappers (two-stage dispatch, single-segment guard)
        function testCouplingHelpers(testCase)
            fixedH = struct('fixed', true, 'mu', 0, 'Sigma', 1);
            cases = { ...  % prior, has_mrf, has_free_hierarchy, is_two_stage, needs_single_segment
                [],                                                  false, false, false, false; ...
                struct(),                                            false, false, false, false; ...
                struct('hierarchical', struct()),                    false, true,  false, true;  ...
                struct('hierarchical', true),                        false, true,  false, true;  ...
                struct('hierarchical', false),                       false, false, false, false; ...
                struct('hierarchical', fixedH),                      false, false, false, false; ...
                struct('hierarchical', fixedH, 'mrf', struct()),     true,  false, false, true;  ...
                struct('hierarchical', struct(), 'mrf', struct()),   true,  true,  true,  true;  ...
                struct('hierarchical', struct(), 'mrf', false),      false, true,  false, true};
            for k = 1:size(cases,1)
                f = struct('prior', {cases{k,1}});
                testCase.verifyEqual([mcmc_bayes.has_mrf(f) mcmc_bayes.has_free_hierarchy(f) mcmc_bayes.is_two_stage(f) ...
                                      mcmc_bayes.needs_single_segment(f)], [cases{k,2:5}], sprintf('case %d', k));
            end
            testCase.verifyFalse(mcmc_bayes.needs_single_segment(struct()));
        end

        %% no kept samples (burn-in >= iterations, as in a model wrapper's memory probe) must not error
        function testZeroKeptSamples(testCase)
            gacelletest.assumeGPU(testCase);
            [y, mask, te] = McmcBayesLegacyTest.r2starData();
            f = struct('solver','mcmc','algorithm','MH','iteration',100,'burnin',10000,'thinning',10,'metric',{{'mean','std'}}, ...
                       'mcmcClass','mcmc_bayes','likelihood','marginal_S0noise','S0Param','M0', ...
                       'parameterTransform',{{'linear','sigmoid','linear'}},'adaptStepSize',true,'repetition',2);
            out = gpuR2starMapping(te).estimate(y, mask, f);                                   % flat
            testCase.verifyEqual(size(out.posterior.R2star, 2), 0);
            f.prior = struct('hierarchical', struct('params', {{'R2star'}}));
            out = gpuR2starMapping(te).estimate(y, mask, f);                                   % free hierarchical
            testCase.verifyTrue(all(isfinite(out.hyper.mean.mu)));
            f.prior.mrf = struct('mode','3d');
            out = gpuR2starMapping(te).estimate(repmat(y,[1 1 2 1]), repmat(mask,[1 1 2]), f); % two-stage
            testCase.verifyEqual(size(out.posterior.R2star, 2), 0);
        end

        %% forward-model output size check
        function testForwardSizeCheck(testCase)
            testCase.verifyError(@() mcmc_bayes.check_forward_size(zeros(1,5), 16, 5), 'mcmc_bayes:forwardSize');
            testCase.verifyError(@() mcmc_bayes.check_forward_size(zeros(16,4), 16, 5), 'mcmc_bayes:forwardSize');
            testCase.verifyError(@() mcmc_bayes.check_forward_size(zeros(16,5,2), 16, 5), 'mcmc_bayes:forwardSize');
            mcmc_bayes.check_forward_size(zeros(16,5), 16, 5);         % no error
            % end to end: 3-D single-slice data [nx,ny,Nb] is read as a volume with one measurement
            gacelletest.assumeGPU(testCase);
            addpath(fullfile(fileparts(mfilename('fullpath')), 'validation', 'mcmc_bayes'));   % ivim_fwd
            b  = [0 0.1 0.3 0.6 0.9];
            f  = struct('modelParams', {{'S0','D','F','Dstar'}}, 'lb', [0 0.1 0 1], 'ub', [2 3 0.5 50], ...
                        'xStepSize', [0.01 0.02 0.01 1], 'algorithm', 'MH', 'likelihood', 'marginal_S0noise', 'S0Param', 'S0', ...
                        'parameterTransform', 'sigmoid', 'iteration', 20, 'burnin', 10, 'thinning', 1, 'metric', {{'mean'}});
            x0 = struct('S0', ones(3,4), 'D', ones(3,4), 'F', 0.1*ones(3,4), 'Dstar', 10*ones(3,4));
            y3 = rand(3, 4, numel(b)) + 0.5;
            testCase.verifyError(@() mcmc_bayes().optimisation(y3, true(3,4), [], x0, f, @ivim_fwd, b), 'mcmc_bayes:forwardSize');
            y4 = reshape(y3, [3 4 1 numel(b)]);
            out = mcmc_bayes().optimisation(y4, true(3,4), [], x0, f, @ivim_fwd, b);
            testCase.verifyTrue(all(isfinite(out.mean.D(:))));
        end

        %% Phase 4: GPU runs of the MRF prior
        function testMrfRuns(testCase, updateScheme)
            gacelletest.assumeGPU(testCase);
            [yy, mask, x0, f, fwd] = McmcBayesUnitTest.linGaussGrid([5 4 3], 2);
            f.updateScheme = updateScheme; f.checkCache = true; f.repetition = 2; f.overdisp = 0.01;
            f.adaptStepSize = true; f.adaptInterval = 20;
            cfgs = {struct('potential','l1'), struct('potential','huber','mode','2d','radius',2), ...
                    struct('potential','quadratic','mode','3d','radius',1,'connectivity','full')};
            for k = 1:numel(cfgs)
                f.prior.mrf = cfgs{k};
                out = mcmc_bayes().optimisation(yy, mask, [], x0, f, fwd);
                cc  = out.diagnostics.cacheCheck;
                testCase.verifyEqual(cc.inactiveMoved, 0, 'a voxel outside the active colour moved');
                testCase.verifyLessThanOrEqual(max([cc.loglik cc.logprior cc.logjac]), 1e-5);
                s   = out.settings.mrf;
                nbr = mcmc_bayes.build_neighbours(find(mask), size(mask), s.mode, s.radius, s.connectivity);
                testCase.verifyEqual(s.Nedges, nnz(nbr)/2);
                testCase.verifyEqual(s.Kneighbours, size(nbr,1));
                testCase.verifyEqual(s.W, 1./sqrt(diag(f.prior.hierarchical.Sigma)));
                testCase.verifyEqual(out.settings.prior.mrf, s);
                testCase.verifyTrue(all(isfinite(out.mean.u1(:))));
                acc = reshape(out.diagnostics.acceptance, numel(mask), []);
                testCase.verifyTrue(all(acc(mask(:),:) > 0, 'all'));
            end
            testCase.verifyEqual(out.settings.mrf.Ncolours, 8);
            % test-only simultaneous update runs and is labelled
            f.mrfUpdate = 'simultaneous'; f.prior.mrf = struct();
            out = mcmc_bayes().optimisation(yy, mask, [], x0, f, fwd);
            testCase.verifyTrue(contains(out.settings.mrf.update, 'TEST ONLY'));
            testCase.verifyEqual(out.settings.mrf.NcoloursUsed, 1);
            % memory guard
            f.mrfUpdate = 'chromatic'; f.prior.mrf = struct('maxGPUMemory', 1);
            testCase.verifyError(@() mcmc_bayes().optimisation(yy, mask, [], x0, f, fwd), 'mcmc_bayes:mrfMemory');
        end

        %% Phase 4b: forward model on the active colour only
        function testMrfSubsetForwardUsed(testCase, updateScheme)
            % used for lingauss_fwd (gaussian, known noise) and ivim_fwd (marginal_S0noise, per-voxel m);
            % caches equal a fresh recompute after the colour steps, inactive voxels never move
            gacelletest.assumeGPU(testCase);
            addpath(fullfile(fileparts(mfilename('fullpath')), 'validation', 'mcmc_bayes'));
            [yy, mask, x0, f, ~] = McmcBayesUnitTest.linGaussGrid([6 5 3], 2);
            rng(3); x0.u1 = 0.5 + 0.3*randn(size(mask)); x0.u2 = 0.5 + 0.3*randn(size(mask));   % non-trivial start
            A   = McmcBayesUnitTest.linGaussA(2);
            fwd = @(p) lingauss_fwd(p, A);
            f.updateScheme = updateScheme; f.checkCache = true; f.adaptStepSize = true; f.adaptInterval = 20;
            for mrfCfg = {struct('potential','l1'), struct('potential','huber','mode','2d','radius',2)}
                f.prior.mrf = mrfCfg{1};
                out = testCase.verifyWarningFree(@() mcmc_bayes().optimisation(yy, mask, [], x0, f, fwd));
                sfi = out.settings.mrf.subsetForward;
                testCase.verifyTrue(sfi.requested && sfi.used, sfi.reason);
                testCase.verifyLessThanOrEqual(sfi.maxRelDiff, 1e-6);
                testCase.verifyTrue(contains(out.settings.mrf.cost, 'active colour only'));
                cc  = out.diagnostics.cacheCheck;
                testCase.verifyEqual(cc.inactiveMoved, 0, 'a voxel outside the active colour moved');
                testCase.verifyLessThanOrEqual(max([cc.loglik cc.logprior cc.logjac]), 1e-5);
                testCase.verifyEqual(cc.Ncheck, f.iteration);
            end
            % IVIM, marginal_S0noise with per-voxel weights (zeros: per-voxel m), log/sigmoid transforms
            [y, maskI, x0I, g, fwdI] = McmcBayesUnitTest.ivimGrid();
            rng(4); w = 0.5 + rand(size(y)); w(rand(size(y)) < 0.05) = 0;
            g.updateScheme = updateScheme; g.checkCache = true;
            g.prior.mrf = struct('potential','l1','tau',1,'mode','2d','radius',2);
            rng(11); parallel.gpu.rng(11);
            outT = mcmc_bayes().optimisation(y, maskI, w, x0I, g, fwdI);
            sfi = outT.settings.mrf.subsetForward;
            testCase.verifyTrue(sfi.used, sfi.reason);
            testCase.verifyTrue(sfi.bitwise, 'ivim_fwd is elementwise: bitwise subset expected');
            cc  = outT.diagnostics.cacheCheck;
            testCase.verifyEqual(cc.inactiveMoved, 0);
            testCase.verifyLessThanOrEqual(max([cc.loglik cc.logprior cc.logjac]), 1e-5);
            % same seed, subsetForward = false: identical chain for the elementwise ivim_fwd
            g.prior.mrf.subsetForward = false;
            rng(11); parallel.gpu.rng(11);
            outF = mcmc_bayes().optimisation(y, maskI, w, x0I, g, fwdI);
            testCase.verifyFalse(outF.settings.mrf.subsetForward.used);
            testCase.verifyEqual(outT.posterior, outF.posterior);
        end

        function testMrfSubsetForwardFallback(testCase)
            % FWD with a voxel-dimensioned varargin input: the subset is detected and not used
            gacelletest.assumeGPU(testCase);
            [yy, mask, x0, f, ~] = McmcBayesUnitTest.linGaussGrid([6 5 3], 2);
            rng(3); x0.u1 = 0.5 + 0.3*randn(size(mask)); x0.u2 = 0.5 + 0.3*randn(size(mask));
            A   = McmcBayesUnitTest.linGaussA(2);
            Nv  = nnz(mask);
            sc  = gpuArray(single(0.5 + rand(1, Nv)));             % per-voxel scale (voxel dimension)
            f.prior.mrf = struct('potential','l1','tau',1);
            f.checkCache = true;
            cases = { ...
                {@(p, s) McmcBayesUnitTest.linGaussFwd(p, A) .* s(1:numel(p.u1)), 'differs'}, ...   % accepts any Nv, wrong columns
                {@(p, s) McmcBayesUnitTest.linGaussFwd(p, A) .* s, 'errors'}, ...                    % implicit expansion fails
                {@(p, s) McmcBayesUnitTest.linGaussFwd(p, A) .* ones(1, numel(p.u1)) + 0*s(1) + ...
                         0*McmcBayesUnitTest.assertNv(p, numel(s)), 'errors'}, ...                  % fixed voxel count
                {@(p, s) repmat(McmcBayesUnitTest.linGaussFwd(p, A), 1, numel(s)/numel(p.u1)), 'size'} };  % [m, Nv] always (errors if not integer)
            for k = 1:numel(cases)
                fwdK = @(p) cases{k}{1}(p, sc);
                rng(12); parallel.gpu.rng(12);
                outT = testCase.verifyWarning(@() mcmc_bayes().optimisation(yy, mask, [], x0, f, fwdK), 'mcmc_bayes:subsetForwardFallback', cases{k}{2});
                sfi  = outT.settings.mrf.subsetForward;
                testCase.verifyTrue(sfi.requested && ~sfi.used, cases{k}{2});
                testCase.verifyNotEmpty(sfi.reason);
                testCase.verifyTrue(contains(outT.settings.mrf.cost, 'all voxels'));
                testCase.verifyEqual(outT.diagnostics.cacheCheck.inactiveMoved, 0);
                % the fallback is the full evaluation: bitwise identical to subsetForward = false
                g = f; g.prior.mrf.subsetForward = false;
                rng(12); parallel.gpu.rng(12);
                outF = mcmc_bayes().optimisation(yy, mask, [], x0, g, fwdK);
                testCase.verifyEqual(outT.posterior, outF.posterior, cases{k}{2});
                testCase.verifyEqual(outT.diagnostics, outF.diagnostics, cases{k}{2});
            end
            testCase.verifyTrue(contains(sfi.reason, 'size') || contains(sfi.reason, 'errors'));
            % the helper directly: reasons per failure type
            p   = struct('u1', gpuArray(single([1 2 3 4])), 'u2', gpuArray(single([1 1 1 1])));
            pS  = {struct('u1', p.u1([1 3]), 'u2', p.u2([1 3])), struct('u1', p.u1([2 4]), 'u2', p.u2([2 4]))};
            idx = {[1 3], [2 4]};
            info = mcmc_bayes.check_subset_forward(@(q) q.u1 + q.u2, {}, p, pS, idx, 4, 1e-6);
            testCase.verifyTrue(info.used && info.bitwise);
            testCase.verifyEqual(info.maxRelDiff, 0);
            info = mcmc_bayes.check_subset_forward(@(q, s) q.u1 .* s(1:numel(q.u1)), {gpuArray(single([1 2 3 4]))}, p, pS, idx, 4, 1e-6);
            testCase.verifyFalse(info.used); testCase.verifyTrue(contains(info.reason, 'differs'));
            info = mcmc_bayes.check_subset_forward(@(q, s) q.u1 .* s, {gpuArray(single([1 2 3 4]))}, p, pS, idx, 4, 1e-6);
            testCase.verifyFalse(info.used); testCase.verifyTrue(contains(info.reason, 'errors'));
            info = mcmc_bayes.check_subset_forward(@(q) q.u1 .* [1 1 1 1], {}, p, pS, idx, 4, 1e-6);
            testCase.verifyFalse(info.used); testCase.verifyTrue(contains(info.reason, 'errors') || contains(info.reason, 'size'));
            info = mcmc_bayes.check_subset_forward(@(q) q.u1 + 1e-5*numel(q.u1), {}, p, pS, idx, 4, 1e-6);   % tiny column coupling
            testCase.verifyFalse(info.used); testCase.verifyTrue(contains(info.reason, 'differs'));
            info = mcmc_bayes.check_subset_forward(@(q) q.u1 .* (1 + 1e-7*(numel(q.u1) == 2)), {}, p, pS, idx, 4, 1e-6);   % within tolerance
            testCase.verifyTrue(info.used); testCase.verifyFalse(info.bitwise);
            % simultaneous (TEST ONLY) and disabled: not used, recorded, no warning
            g = f; g.mrfUpdate = 'simultaneous';
            out = testCase.verifyWarningFree(@() mcmc_bayes().optimisation(yy, mask, [], x0, g, @(p) McmcBayesUnitTest.linGaussFwd(p, A)));
            testCase.verifyFalse(out.settings.mrf.subsetForward.used);
            testCase.verifyTrue(contains(out.settings.mrf.subsetForward.reason, 'simultaneous'));
        end

        function testRunTwoStage(testCase)
            gacelletest.assumeGPU(testCase);
            [yy, mask, x0, f, fwd] = McmcBayesUnitTest.linGaussGrid([6 5 3], 2);
            f.prior.hierarchical = struct();                       % free (stage 1)
            f.prior.mrf = struct('potential','l1','tau',2);
            out = mcmc_bayes().run_two_stage(yy, mask, [], x0, f, fwd);
            eb  = out.settings.empiricalBayes;
            testCase.verifyTrue(contains(eb.scheme, 'NOT the full joint posterior'));
            testCase.verifyEqual(eb.muHat, out.stage1.hyper.mean.mu);
            testCase.verifyEqual(eb.SigmaHat, out.stage1.hyper.mean.Sigma);
            testCase.verifyTrue(out.hyper.fixed);
            testCase.verifyEqual(out.hyper.mean.Sigma, eb.SigmaHat);
            testCase.verifyEqual(out.settings.mrf.W, 1./sqrt(diag(eb.SigmaHat)));
            testCase.verifyEqual(eb.Nstage1, nnz(mask));
            testCase.verifyEqual(size(out.posterior.u1, 1), nnz(mask));
            testCase.verifyTrue(all(isfinite(out.mean.u2(:))));
            % stage 1 on a subset
            f.prior.hierarchical = struct('subsetFraction', 0.5);
            rng(5);
            out = mcmc_bayes().run_two_stage(yy, mask, [], x0, f, fwd);
            testCase.verifyEqual(out.settings.empiricalBayes.Nstage1, ceil(0.5*nnz(mask)));
            testCase.verifyEqual(numel(out.stage1.voxelIndex), ceil(0.5*nnz(mask)));
            testCase.verifyEqual(size(out.posterior.u1, 1), nnz(mask));
        end

        %% Adaptive covariance (fitting.adaptCovariance)
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
                    [L, ok] = mcmc_bayes.chol_batch(devs{kd}(C));
                    L = gather(L);
                    testCase.verifyTrue(all(gather(ok)));
                    testCase.verifyLessThanOrEqual(max(abs(L - Lref), [], 'all'), 1e-10 * max(abs(Lref), [], 'all'), sprintf('d = %d', d));
                    testCase.verifyEqual(triu(L(:,:,1), 1), zeros(d));      % lower triangular
                end
                % indefinite and NaN pages are flagged, the others are not affected
                C = repmat(eye(3), 1, 1, 3); C(:,:,2) = [1 2 0; 2 1 0; 0 0 1]; C(1,1,3) = NaN;
                [L, ok] = mcmc_bayes.chol_batch(devs{kd}(C));
                testCase.verifyEqual(gather(ok), [true false false]);
                testCase.verifyEqual(gather(L(:,:,1)), eye(3));
            end
        end

        function testAcovWelford(testCase)
            rng(62); d = 4; Nv = 30; T = 500;
            A = randn(d); X = 3 + pagemtimes(A, randn(d, T, Nv)) .* reshape(logspace(-3, 2, Nv), 1, 1, Nv);   % [d,T,Nv]
            m = zeros(d, Nv); M2 = zeros(d, d, Nv); n = 0;
            for t = 1:T
                [m, M2, n] = mcmc_bayes.welford_update(m, M2, n, squeeze(X(:,t,:)));
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
                for t = 1:T; [mG, M2G, nG] = mcmc_bayes.welford_update(mG, M2G, nG, gpuArray(squeeze(Xs(:,t,:)))); end
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
            [L, valid] = mcmc_bayes.acov_refresh(M2, n, accN, Lprev, false(1, N), 1e-6, 1e-12);
            Creg = C(:,:,1) + 1e-6*diag(diag(C(:,:,1))) + (1e-12*2 + realmin('double'))*eye(2);
            testCase.verifyEqual(double(L(:,:,1)), chol(Creg, 'lower'), 'RelTol', 1e-6);   % single output
            testCase.verifyEqual(L(:,:,2), Lprev(:,:,2));        % zero variance: previous factor kept
            testCase.verifyEqual(L(:,:,3), Lprev(:,:,3));        % too few acceptances: previous factor kept
            testCase.verifyEqual(valid, [true false false]);
            [~, valid] = mcmc_bayes.acov_refresh(M2, n, accN, Lprev, [false true false], 1e-6, 1e-12);
            testCase.verifyEqual(valid, [true true false]);      % once valid, stays valid (older factor kept)
            % the proposal increment is L*z per voxel
            Lp = randn(3, 3, 4); z = randn(3, 4);
            s  = mcmc_bayes.acov_step(Lp, z);
            for v = 1:4; testCase.verifyEqual(s(:,v), Lp(:,:,v)*z(:,v), 'AbsTol', 1e-12); end
        end

        function testAdaptCovarianceOptionErrors(testCase)
            f = struct('adaptCovariance', true, 'adaptStepSize', true, 'updateScheme', 'componentwise');
            testCase.verifyError(@() mcmc_bayes.check_set_default_bayes(f), 'mcmc_bayes:adaptCovariance');
            testCase.verifyError(@() mcmc_bayes().optimisation([], [], [], [], f, []), 'mcmc_bayes:adaptCovariance');
            f = struct('adaptCovariance', true);                 % adaptStepSize defaults to false
            testCase.verifyError(@() mcmc_bayes.check_set_default_bayes(f), 'mcmc_bayes:adaptCovariance');
            f = struct('adaptCovariance', true, 'adaptStepSize', false);
            testCase.verifyError(@() mcmc_bayes().optimisation([], [], [], [], f, []), 'mcmc_bayes:adaptCovariance');
            f = struct('adaptCovariance', 'yes', 'adaptStepSize', true);
            testCase.verifyError(@() mcmc_bayes.check_set_default_bayes(f), 'mcmc_bayes:adaptCovariance');
            f = struct('adaptCovariance', [1 1], 'adaptStepSize', true);
            testCase.verifyError(@() mcmc_bayes.check_set_default_bayes(f), 'mcmc_bayes:adaptCovariance');
            % valid settings and defaults
            f = mcmc_bayes.check_set_default_bayes(struct('adaptCovariance', 1, 'adaptStepSize', true));
            testCase.verifyTrue(islogical(f.adaptCovariance) && f.adaptCovariance);
            f = mcmc_bayes.check_set_default_bayes(struct());
            testCase.verifyFalse(f.adaptCovariance);
            testCase.verifyTrue(mcmc_bayes.isLegacy(struct('adaptCovariance', false)));
        end

        function testAdaptCovarianceCacheConsistency(testCase)
            % caches with the adaptive-covariance proposal: gaussian + transforms, free hierarchical
            % prior, MRF (full and subsetForward colour steps)
            gacelletest.assumeGPU(testCase);
            [y, mask, w, pars0, fitting, obj] = McmcBayesUnitTest.r2starSetup();
            g = fitting;
            g.lb = [0; 0; 0.001]; g.ub = [2; Inf; 0.1]; g.parameterTransform = {'sigmoid','log','linear'};
            g.adaptStepSize = true; g.adaptInterval = 10; g.adaptCovariance = true; g.checkCache = true;
            g.iteration = 300; g.burnin = 150; g.repetition = 2; g.overdisp = 0.01;
            for hierCfg = {[], struct('hyperprior','niw')}
                g.prior = []; g.ub(2) = 200;                    % 'log' outside the hierarchy: finite ub
                if ~isempty(hierCfg{1}); g.prior.hierarchical = hierCfg{1}; g.ub(2) = Inf; end
                out = mcmc_bayes().optimisation(y, mask, w, pars0, g, @obj.FWD, 'mcmc', g);
                cc  = out.diagnostics.cacheCheck;
                testCase.verifyEqual(cc.Ncheck, g.iteration*g.repetition);
                testCase.verifyLessThanOrEqual(max([cc.loglik cc.logprior cc.logjac]), 1e-5);
                ac  = out.diagnostics.adaptCovariance;
                testCase.verifyEqual(ac.switchIteration, 50);    % first multiple of 10 with k - 20 >= 30
                testCase.verifyEqual(size(ac.proposalCov), [size(mask,1:3) 3 3 2]);
                testCase.verifyEqual(ac.params, {'M0','R2star','noise'});
                testCase.verifyTrue(out.settings.adaptCovariance);
                testCase.verifyGreaterThanOrEqual(min(out.posterior.noise(:)), single(g.lb(3)));
                testCase.verifyLessThanOrEqual(max(out.posterior.noise(:)), single(g.ub(3)));
                % stepSize = sqrt(diag(proposalCov))
                P = reshape(ac.proposalCov, [], 3, 3, 2);
                testCase.verifyEqual(reshape(out.diagnostics.stepSize.R2star, [], 2), reshape(sqrt(P(:,2,2,:)), [], 2), 'RelTol', 1e-6);
            end
            % MRF, subsetForward true and false, linear Gaussian grid (joint)
            [yy, mask, x0, f, ~] = McmcBayesUnitTest.linGaussGrid([6 5 3], 2);
            rng(3); x0.u1 = 0.5 + 0.3*randn(size(mask)); x0.u2 = 0.5 + 0.3*randn(size(mask));
            A   = McmcBayesUnitTest.linGaussA(2);
            fwd = @(p) McmcBayesUnitTest.linGaussFwd(p, A);
            f.checkCache = true; f.adaptStepSize = true; f.adaptInterval = 10; f.adaptCovariance = true;
            f.iteration = 200; f.burnin = 100;
            for sf = [true false]
                f.prior.mrf = struct('potential','l1','subsetForward', sf);
                out = mcmc_bayes().optimisation(yy, mask, [], x0, f, fwd);
                testCase.verifyEqual(out.settings.mrf.subsetForward.used, sf);
                cc  = out.diagnostics.cacheCheck;
                testCase.verifyEqual(cc.inactiveMoved, 0, 'a voxel outside the active colour moved');
                testCase.verifyLessThanOrEqual(max([cc.loglik cc.logprior cc.logjac]), 1e-5);
                testCase.verifyEqual(out.diagnostics.adaptCovariance.switchIteration, 40);
                v = out.diagnostics.adaptCovariance.valid;
                testCase.verifyGreaterThanOrEqual(mean(double(v(mask))), 0.9);
            end
        end

        function testAdaptCovarianceRuns(testCase)
            % IVIM grid (marginal_S0noise, log/sigmoid rows, fixed hierarchical + MRF): subsetForward on and
            % off give the same chain for the elementwise ivim_fwd with the covariance proposal
            gacelletest.assumeGPU(testCase);
            addpath(fullfile(fileparts(mfilename('fullpath')), 'validation', 'mcmc_bayes'));
            [y, maskI, x0I, g, fwdI] = McmcBayesUnitTest.ivimGrid();
            g.adaptCovariance = true; g.adaptInterval = 10; g.burnin = 100; g.iteration = 200; g.checkCache = true;
            g.prior.mrf = struct('potential','l1','tau',1,'mode','2d','radius',2);
            rng(11); parallel.gpu.rng(11);
            outT = mcmc_bayes().optimisation(y, maskI, [], x0I, g, fwdI);
            testCase.verifyTrue(outT.settings.mrf.subsetForward.used);
            testCase.verifyEqual(outT.diagnostics.adaptCovariance.switchIteration, 50);   % d = 3 (D, F, Dstar)
            testCase.verifyEqual(outT.diagnostics.adaptCovariance.params, {'D','F','Dstar'});
            cc = outT.diagnostics.cacheCheck;
            testCase.verifyEqual(cc.inactiveMoved, 0);
            testCase.verifyLessThanOrEqual(max([cc.loglik cc.logprior cc.logjac]), 1e-5);
            g.prior.mrf.subsetForward = false;
            rng(11); parallel.gpu.rng(11);
            outF = mcmc_bayes().optimisation(y, maskI, [], x0I, g, fwdI);
            testCase.verifyEqual(outT.posterior, outF.posterior);
            testCase.verifyEqual(outT.diagnostics.adaptCovariance.proposalCov, outF.diagnostics.adaptCovariance.proposalCov);
        end

        function testAdaptCovarianceNoSwitch(testCase)
            % burn-in too short for the switch: warning, diagonal proposal throughout (== adaptCovariance false)
            gacelletest.assumeGPU(testCase);
            [yy, mask, x0, f, fwd] = McmcBayesUnitTest.linGaussSetup(20, 2);
            f.adaptStepSize = true; f.adaptInterval = 20; f.burnin = 50;     % warm-up 40, steps at 20, 40 only
            rng(8); parallel.gpu.rng(8);
            outD = mcmc_bayes().optimisation(yy, mask, [], x0, f, fwd);
            f.adaptCovariance = true;
            rng(8); parallel.gpu.rng(8);
            outA = testCase.verifyWarning(@() mcmc_bayes().optimisation(yy, mask, [], x0, f, fwd), 'mcmc_bayes:adaptCovarianceNoSwitch');
            testCase.verifyEqual(outA.posterior, outD.posterior);
            testCase.verifyTrue(isnan(outA.diagnostics.adaptCovariance.switchIteration));
            P = reshape(outA.diagnostics.adaptCovariance.proposalCov, [], 2, 2);
            testCase.verifyEqual(P(:,1,1), outD.diagnostics.stepSize.u1(:).^2, 'RelTol', 1e-6);
            testCase.verifyEqual(P(:,1,2), zeros(20, 1, 'single'));
        end

        function testAdaptCovarianceCorrelatedToy(testCase)
            % flat prior, known noise: posterior N((A'A)^-1 A'y, s^2 (A'A)^-1), correlation -0.995
            gacelletest.assumeGPU(testCase);
            addpath(fullfile(fileparts(mfilename('fullpath')), 'validation', 'mcmc_bayes'));
            rng(5); Nv = 200; m = 6; s = 0.3; d = 2;
            A   = [ones(m,1), 1 + 0.15*linspace(-1,1,m).'];
            P   = s^2 * inv(A.'*A);
            y   = A*randn(d, Nv) + s*randn(m, Nv);
            uhat = (A.'*A) \ (A.'*y);
            yy  = reshape(y.', [Nv 1 1 m]); mask = true(Nv,1);
            f.modelParams = {'u1';'u2';'noise'}; f.lb = [-Inf;-Inf;0]; f.ub = [Inf;Inf;10]; f.xStepSize = [0.2;0.2;0.01];
            f.algorithm = 'MH'; f.iteration = 6000; f.burnin = 2000; f.thinning = 1; f.metric = {'mean'};
            f.fixedParams = struct('noise', s); f.adaptStepSize = true; f.adaptInterval = 50;
            x0  = struct('u1', zeros(Nv,1), 'u2', zeros(Nv,1));
            fwd = @(p) lingauss_fwd(p, A);
            essIt = zeros(1, 2);
            for ac = [false true]
                g = f; g.adaptCovariance = ac;
                rng(1); parallel.gpu.rng(1);
                out = mcmc_bayes().optimisation(yy, mask, [], x0, g, fwd);
                Ns  = size(out.posterior.u1, 2);
                E   = [out.diagnostics.ess.u1(:) out.diagnostics.ess.u2(:)].';          % [2,Nv]
                essIt(ac+1) = median(E(1,:)) / Ns;
                if ~ac; continue; end
                mh  = [mean(out.posterior.u1, 2) mean(out.posterior.u2, 2)].';
                z   = (mh - uhat) ./ sqrt(diag(P) ./ E);
                testCase.verifyGreaterThanOrEqual(mean(z.^2, 2), [0.5; 0.5]);
                testCase.verifyLessThanOrEqual(mean(z.^2, 2), [2; 2]);
                r   = [reshape(double(out.posterior.u1) - uhat(1,:).', 1, []); reshape(double(out.posterior.u2) - uhat(2,:).', 1, [])];
                Ch  = r*r.' / size(r, 2);
                se  = sqrt(2 / sum(E(1,:)));
                testCase.verifyLessThanOrEqual(max(abs(Ch ./ P - 1), [], 'all'), 5*se, sprintf('pooled covariance ./ exact:\n%s', mat2str(Ch./P, 4)));
                acd = out.diagnostics.adaptCovariance;
                testCase.verifyEqual(acd.switchIteration, 150);
                testCase.verifyTrue(all(acd.valid));
                Pc  = squeeze(median(acd.proposalCov, 1));
                testCase.verifyEqual(double(Pc(1,2)/sqrt(Pc(1,1)*Pc(2,2))), P(1,2)/sqrt(P(1,1)*P(2,2)), 'AbsTol', 0.01);
                testCase.verifyGreaterThan(median(out.diagnostics.acceptance(:)), 0.15);
                testCase.verifyLessThan(median(out.diagnostics.acceptance(:)), 0.4);
            end
            gain = essIt(2) / essIt(1);
            testCase.log(matlab.unittest.Verbosity.Terse, sprintf('correlated toy: ESS/iteration u1 %.4f (diagonal) -> %.4f (adaptCovariance), gain %.1f', essIt(1), essIt(2), gain));
            testCase.verifyGreaterThanOrEqual(gain, 5);
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

        % small linear Gaussian toy (y = A u + e, known noise via the test-only fixedParams)
        function [yy, mask, x0, f, fwd] = linGaussSetup(Nv, d)
            rng(40); m = 6; s = 0.5;
            A   = randn(m, d);
            u   = 0.5 + randn(d, Nv);
            y   = (A*u + s*randn(m, Nv)).';
            yy  = reshape(y, [Nv 1 1 m]); mask = true(Nv, 1);
            f.modelParams = [arrayfun(@(k) sprintf('u%d',k), (1:d).', 'UniformOutput', false); {'noise'}];
            f.lb = [-Inf(d,1); 0]; f.ub = [Inf(d,1); 10]; f.xStepSize = [0.2*ones(d,1); 0.01];
            f.algorithm = 'MH'; f.iteration = 300; f.burnin = 100; f.thinning = 5; f.metric = {'mean'};
            f.fixedParams = struct('noise', s);
            for k = 1:d; x0.(sprintf('u%d',k)) = zeros(Nv,1); end
            fwd = @(p) McmcBayesUnitTest.linGaussFwd(p, A);
        end

        function s = linGaussFwd(pars, A)
            d = size(A, 2);
            u = zeros(d, numel(pars.u1), 'like', pars.u1);
            for k = 1:d; u(k,:) = pars.(sprintf('u%d',k))(:).'; end
            s = cast(A, 'like', u) * u;
        end

        % design matrix of linGaussGrid (same seed and draw order)
        function A = linGaussA(d)
            rng(42); m = 5;
            dims = [6 5 3]; rand(dims); %#ok<RAND>                % the mask draw of linGaussGrid comes first
            A = randn(m, d);
        end

        % error unless the parameter structure has N voxels (a FWD with a fixed voxel count)
        function z = assertNv(p, N)
            if numel(p.u1) ~= N; error('McmcBayesUnitTest:fixedNv', 'fixed voxel count %d (got %d)', N, numel(p.u1)); end
            z = 0;
        end

        % small IVIM grid (ivim_fwd), marginal_S0noise, hierarchical fixed around the truth
        function [y, mask, x0, f, fwd] = ivimGrid()
            rng(43); dims = [7 6 3];
            mask = rand(dims) > 0.15; Nv = nnz(mask);
            b    = [0 0.02 0.05 0.1 0.2 0.4 0.7 1 1.5 2 3];
            mu   = [log(1); log(0.1/0.9); log(20)]; Sigma = diag([0.1 0.3 0.2].^2);
            u    = mu + chol(Sigma,'lower')*randn(3, Nv);
            p    = struct('S0', 1 + 0.05*randn(1, Nv), 'D', exp(u(1,:)), 'F', 1./(1+exp(-u(2,:))), 'Dstar', exp(u(3,:)));
            s    = ivim_fwd(p, b) + 0.02*randn(numel(b), Nv);
            y    = zeros(numel(mask), numel(b)); y(mask(:), :) = s.';
            y    = reshape(y, [dims numel(b)]);
            o    = ones(dims);
            x0   = struct('S0', o, 'D', o, 'F', 0.1*o, 'Dstar', 20*o, 'noise', 0.02*o);
            f.modelParams = {'S0';'D';'F';'Dstar';'noise'};
            f.lb = [0; 0; 0; 0; 0.001]; f.ub = [2; Inf; 1; Inf; 1];
            f.xStepSize = [0.01; 0.05; 0.01; 2; 0.001];
            f.parameterTransform = {'linear','log','sigmoid','log','linear'};
            f.likelihood = 'marginal_S0noise'; f.S0Param = 'S0';
            f.algorithm = 'MH'; f.iteration = 150; f.burnin = 50; f.thinning = 5; f.metric = {'mean'};
            f.adaptStepSize = true; f.adaptInterval = 25;
            f.prior.hierarchical = struct('fixed', true, 'mu', mu, 'Sigma', Sigma);
            fwd  = @(pp) ivim_fwd(pp, b);
        end

        % log NIW(mu, Sigma | m, k, Psi, nu), all (mu,Sigma)-dependent terms
        function lp = logNIW(mu, Sigma, m, k, Psi, nu)
            d  = numel(mu); r = mu - m;
            lp = -0.5*log(det(Sigma)) - 0.5*k*(r.'/Sigma)*r - (nu+d+1)/2*log(det(Sigma)) - 0.5*trace(Psi/Sigma);
        end

        % linear Gaussian toy on a 3D grid with holes, fixed hierarchical prior (MRF tests)
        function [yy, mask, x0, f, fwd] = linGaussGrid(dims, d)
            rng(42); m = 5; s = 0.5;
            mask = rand(dims) > 0.2;
            Nv  = nnz(mask);
            A   = randn(m, d);
            mu  = 0.5*ones(d,1); Sigma = 0.3*eye(d) + 0.05;
            u   = mu + chol(Sigma,'lower')*randn(d, Nv);
            y   = zeros(numel(mask), m);
            y(mask(:), :) = (A*u + s*randn(m, Nv)).';
            yy  = reshape(y, [dims m]);
            f.modelParams = [arrayfun(@(k) sprintf('u%d',k), (1:d).', 'UniformOutput', false); {'noise'}];
            f.lb = [-Inf(d,1); 0]; f.ub = [Inf(d,1); 10]; f.xStepSize = [0.2*ones(d,1); 0.01];
            f.algorithm = 'MH'; f.iteration = 200; f.burnin = 50; f.thinning = 5; f.metric = {'mean'};
            f.fixedParams = struct('noise', s);
            f.prior.hierarchical = struct('fixed', true, 'mu', mu, 'Sigma', Sigma);
            for k = 1:d; x0.(sprintf('u%d',k)) = zeros(dims); end
            fwd = @(p) McmcBayesUnitTest.linGaussFwd(p, A);
        end

        % symmetric positive edge weights in the layout of nbr (0 for absent neighbours)
        function w = symmetricEdgeWeights(nbr)
            [K, Nv] = size(nbr);
            w = zeros(K, Nv);
            [k, v] = find(nbr > 0);
            n = double(nbr(sub2ind([K Nv], k, v)));
            a = min(v, n); b = max(v, n);
            w(sub2ind([K Nv], k, v)) = 0.5 + mod(a*7919 + b*104729, 97)/97;   % depends on the unordered pair only
        end

        % reference potential (independent of mcmc_bayes.mrf_rho)
        function r = rhoRef(x, potential, delta)
            switch potential
                case 'l1';        r = abs(x);
                case 'quadratic'; r = x.^2/2;
                case 'huber'
                    if abs(x) <= delta; r = x^2/(2*delta); else; r = abs(x) - delta/2; end
            end
        end

        % brute-force Phi_MRF over all edges, each counted once (double, loops)
        function phi = bruteForcePhi(u, nbr, w, W, tau, potential, delta)
            phi = 0;
            [K, Nv] = size(nbr);
            for v = 1:Nv
                for k = 1:K
                    n = double(nbr(k,v));
                    if n > v
                        for p = 1:size(u,1)
                            phi = phi + W(p)/tau * w(k,v) * McmcBayesUnitTest.rhoRef(u(p,v) - u(p,n), potential, delta(p));
                        end
                    end
                end
            end
        end

        % sum of |terms| entering the local differences (scale for the single-precision tolerance)
        function S = localScale(uNewA, uOldA, uAll, nbrA, wA, coef, delta, potential)
            S = 0;
            for c = 1:size(nbrA, 2)
                for k = 1:size(nbrA, 1)
                    for p = 1:size(uNewA, 1)
                        un = uAll(p, nbrA(k,c));
                        S  = S + coef(p) * wA(k,c) * (McmcBayesUnitTest.rhoRef(uNewA(p,c) - un, potential, delta(p)) + ...
                                                     McmcBayesUnitTest.rhoRef(uOldA(p,c) - un, potential, delta(p)));
                    end
                end
            end
        end

        function fitting = explicitDefaults()
            fitting.parameterTransform  = 'linear';
            fitting.likelihood          = 'gaussian';
            fitting.S0Param             = '';
            fitting.updateScheme        = 'joint';
            fitting.adaptStepSize       = false;
            fitting.adaptInterval       = 50;
            fitting.adaptTarget         = [];
            fitting.adaptCovariance     = false;
            fitting.overdisp            = 0;
            fitting.prior               = [];
        end
    end
end
