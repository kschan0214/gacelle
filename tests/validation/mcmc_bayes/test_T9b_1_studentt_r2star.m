%% test_T9b_1_studentt_r2star.m
%
% T9b.1 (Phase 9b): characterisation, NO pass/fail.
%
% Student-t population prior vs the Normal (K = 1) and the Gaussian mixture (K = 2) on the R2* phantom of
% Phase 6a/7: QSM Challenge 2.0 based R2* and M0 maps (Marques et al. 2021) resampled to 48 x 48 x 24,
% 12 echoes (te = 3:3:36 ms), Gaussian noise at SNR 10 (build_r2star_phantom.m, seed 1). Under the K = 1
% Normal prior the high-R2* nuclei (globus pallidus, dentate) are pulled towards the WM/GM population
% (Phase 6a: GP coverage 0.11); K = 2 recovered much of it (Phase 7: GP coverage 0.74). The t prior
% (nu = 4) discounts such voxels without choosing groups.
%
% Arms (same data, same sampler settings as run_7_r2star.m, seed 11 per arm):
%   D     prior.hierarchical K = 1, Normal ('niw')
%   D_K2  prior.hierarchical K = 2 (Gaussian mixture)
%   D_t   prior.hierarchical distribution = 't', nu = 4, K = 1 ('niw')
%   all on R2star only; likelihood 'marginal_S0noise' (S0Param M0), transforms linear/sigmoid/linear,
%   adaptStepSize, 4 repetitions (overdisp 0.01), gpuR2starMapping.estimate with mcmcClass mcmc_bayes.
%   With the Phase 6a results file present (full run only), its flat-prior arm C is shown for reference.
%
% Readouts: overall and per label (2 GP, 5 dentate, 11, 10, 9 GM, 8 WM, 7 thalamus) RMSE and bias of the
%   posterior median and the 90% coverage (5%/95% sample quantiles); voxel R-hat; hyperparameter R-hat
%   (mu, Sigma; K = 2: also pi); for D_t the posterior mean of lambda (out.hyper.lambda) per label.
%
% Data: the phantom is built from the QSM Challenge ground truth, which must not be redistributed, so
%   nothing is stored in the repository. The environment variable MCMC_BAYES_PHASE6_DIR must point to the
%   development folder with build_r2star_phantom.m (and, optionally, results_6a_part2_12echo_snr10.mat);
%   the default below is Kwok's LOCAL path.
% Sizes: full = 5000 iterations, burn-in 2500 (as Phase 7, ~3 min per arm on an A40). The environment
%   variable MCMC_BAYES_T9B_PILOT = 1 runs a pilot (600 iterations, burn-in 300) that only checks the script.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T9b_1_studentt_r2star
%
% Kwok-Shing Chan @ MGH
% Date created: 29 September 2026
%

clearvars; tStart = tic;

isPilot = strcmp(getenv('MCMC_BAYES_T9B_PILOT'), '1');
devDir  = getenv('MCMC_BAYES_PHASE6_DIR');
if isempty(devDir)
    devDir = '/autofs/space/virtuoso_001/users/kwokshing/project/gacelle/development/mcmc_bayes_phase6';   % LOCAL path (Kwok)
end
addpath(devDir);

P = build_r2star_phantom([48 48 24], 10, 1, (3:3:36)*1e-3);
base = struct('solver','mcmc','algorithm','MH','metric',{{'mean','median','std'}},'mcmcClass','mcmc_bayes', ...
              'likelihood','marginal_S0noise','S0Param','M0','parameterTransform',{{'linear','sigmoid','linear'}}, ...
              'adaptStepSize',true,'iteration',5000,'burnin',2500,'thinning',1,'repetition',4,'overdisp',0.01);
if isPilot; base.iteration = 600; base.burnin = 300; end
x  = @(u) 0.1 + 199.9 ./ (1 + exp(-u));             % R2star from u (sigmoid on [0.1, 200])
m  = P.mask; idx = find(m); t = P.truth.R2star; lab = P.label;
labs = [2 5 11 10 9 8 7]; labName = {'GP','dentate','11','10','GM','WM','thal'};

fprintf('T9b.1 Student-t prior on the R2* phantom (characterisation, no pass/fail)%s\n', repmat(' [PILOT]', 1, isPilot));
fprintf('48x48x24, %d voxels, 12 echoes, SNR 10; iteration %d, burn-in %d, 4 repetitions; seed 11 per arm\n\n', nnz(m), base.iteration, base.burnin);

arms = {'D','D_K2','D_t'};
R = struct();
for a = 1:numel(arms)
    f = base;
    switch arms{a}
        case 'D';    f.prior = struct('hierarchical', struct('params', {{'R2star'}}));
        case 'D_K2'; f.prior = struct('hierarchical', struct('params', {{'R2star'}}, 'K', 2));
        case 'D_t';  f.prior = struct('hierarchical', struct('params', {{'R2star'}}, 'distribution', 't', 'nu', 4));
    end
    rng(11); parallel.gpu.rng(11);
    tt = tic; [~, out] = evalc('gpuR2starMapping(P.te).estimate(P.data, P.mask, f);'); S.t = toc(tt);
    xs = out.posterior.R2star; xs = reshape(xs, size(xs,1), [], f.repetition);
    S.median = out.median.R2star;
    S.q5  = zeros(size(m)); S.q5(idx)  = sample_quantile(reshape(xs, size(xs,1), []), 0.05, 2);
    S.q95 = zeros(size(m)); S.q95(idx) = sample_quantile(reshape(xs, size(xs,1), []), 0.95, 2);
    S.rhat = mcmc_bayes.rhat(xs); S.hyper = out.hyper;
    R.(arms{a}) = S; clear S out xs
    fprintf('%s done (%.1f min)\n', arms{a}, R.(arms{a}).t/60);
end

% flat-prior reference (Phase 6a arm C), full run only
refFile = fullfile(devDir, 'results_6a_part2_12echo_snr10.mat');
if ~isPilot && exist(refFile, 'file')
    L = load(refFile, 'Rr');
    if isequal(L.Rr.P.data, P.data); R.C_flat = L.Rr.C; arms = [{'C_flat'}, arms]; end
end

% ---- report ----
fprintf('\n%-8s %6s %7s %6s |', 'arm', 'RMSE', '|e|>10', 'cov90'); fprintf(' %-18s', labName{:}); fprintf('\n');
fprintf('%-8s %6s %7s %6s |', '', '', '', ''); c = repmat({'RMSE bias cov'}, 1, numel(labs)); fprintf(' %-18s', c{:}); fprintf('\n');
for a = 1:numel(arms)
    S = R.(arms{a});
    e = S.median(m) - t(m); cv = t(m) >= S.q5(m) & t(m) <= S.q95(m);
    fprintf('%-8s %6.2f %7.4f %6.3f |', arms{a}, sqrt(mean(e.^2)), mean(abs(e) > 10), mean(cv));
    for Lb = labs
        mm = m & lab == Lb; ee = S.median(mm) - t(mm); cc = mean(t(mm) >= S.q5(mm) & t(mm) <= S.q95(mm));
        fprintf(' %5.2f %+6.2f %4.2f ', sqrt(mean(ee.^2)), mean(ee), cc);
    end
    fprintf('\n');
end
fprintf('\n# voxels per label: %s\n', mat2str(arrayfun(@(Lb) nnz(m & lab == Lb), labs)));
for a = {'D','D_K2','D_t'}
    S = R.(a{1}); H = S.hyper;
    fprintf('%-5s: mu (R2*) %s, sqrt(Sigma) (u) %s', a{1}, mat2str(x(H.mean.mu(:).'), 3), mat2str(sqrt(squeeze(H.mean.Sigma(:)).'), 2));
    if isfield(H, 'rhat')
        fprintf(' | hyper R-hat mu %s, Sigma %s', mat2str(H.rhat.mu(:).', 3), mat2str(H.rhat.Sigma(:).', 3));
        if isfield(H.rhat, 'pi'); fprintf(', pi %s', mat2str(H.rhat.pi(:).', 3)); end
    end
    fprintf(' | voxel R-hat median %.4f, >1.01 %.3f | %.1f min\n', median(S.rhat), mean(S.rhat > 1.01), S.t/60);
end
lam = R.D_t.hyper.lambda;
fprintf('\nD_t posterior mean lambda (prior mean 1; small = treated as an outlier):\n');
for k = 1:numel(labs)
    mm = m & lab == labs(k);
    fprintf('  %-8s median %.3f, 10th pct %.3f\n', labName{k}, median(lam(mm)), sample_quantile(lam(mm), 0.10, 1));
end
fprintf('  all      median %.3f; fraction of voxels with lambda < 0.5: %.3f\n', median(lam(m)), mean(lam(m) < 0.5));
fprintf('\nT9b.1 total time %.1f min\n', toc(tStart)/60);

% sample quantile along dim (linear interpolation between order statistics, h = p(n-1)+1)
function q = sample_quantile(x, p, dim)
x  = sort(x, dim);
n  = size(x, dim);
h  = p*(n - 1) + 1; lo = floor(h); hi = min(lo + 1, n); t = h - lo;
idx = repmat({':'}, 1, ndims(x));
idx{dim} = lo; xl = x(idx{:});
idx{dim} = hi; xh = x(idx{:});
q  = xl + t .* (xh - xl);
end
