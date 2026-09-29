%% test_T9b_2_studentt_meax.m
%
% T9b.2 (Phase 9b): characterisation, NO pass/fail.
%
% Student-t population prior vs the Normal (K = 1) and the Gaussian mixture (K = 2) on the ME-AxCaliberSMT
% phantom of Phase 6/7 (phantom_meax.mat, built by build_meax_phantom.m from in vivo MCMC medians of one
% subject: f, fcsf, DeR, r, R2e on 48 x 48 x 24, 17 shells x 2 TE, spherical means, Gaussian noise).
% Voxel classes as run_meax_k2.m: WM-like (f > 0.3, fcsf <= 0.3), CSF-like (fcsf > 0.3), GM-like (rest).
%
% Arms (same data and sampler settings as run_meax.m / run_meax_k2.m, seed 21 per arm):
%   D     prior.hierarchical on {f, fcsf, DeR, r, R2e}, K = 1 Normal ('niw')
%   D_K2  the same with K = 2 (Gaussian mixture)
%   D_t   the same with distribution = 't', nu = 4 (K = 1, 'niw')
%   likelihood 'marginal_noise', all transforms 'sigmoid', joint updates, adaptStepSize + adaptCovariance,
%   start 'likelihood', 4 repetitions (overdisp 0.01), gpuMEAxCaliberSMT.estimate with mcmcClass mcmc_bayes.
%
% Readouts per class (WM-like / GM-like / CSF-like) for r and R2e (and the other parameters): bias and RMSE of
%   the posterior median, 90% coverage (5%/95% sample quantiles); voxel R-hat; hyperparameter R-hat
%   (mu, Sigma; K = 2: also pi); for D_t the posterior mean of lambda (out.hyper.lambda) per class.
%
% Data: phantom_meax.mat is derived from in vivo data and stays in the development folder (nothing is stored
%   in the repository). MCMC_BAYES_PHASE6_DIR must point to that folder (default below: Kwok's LOCAL path);
%   the class gpuMEAxCaliberSMT lives in AxCaliberSMT/ (on the path via addpath_gacelle).
% Sizes: full = 20000 iterations, burn-in 10000 (as Phase 6/7, ~19-24 min per arm on an A40, ~1 h in total).
%   MCMC_BAYES_T9B_PILOT = 1 runs a pilot (1000 iterations, burn-in 500) that only checks the script.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T9b_2_studentt_meax
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

L  = load(fullfile(devDir, 'phantom_meax.mat'), 'P'); P = L.P; clear L
pn = {'f','fcsf','DeR','r','R2e'}; m = P.mask; idx = find(m);
P.label = 2*m; P.label(m & P.truth.f > 0.3 & P.truth.fcsf <= 0.3) = 1; P.label(m & P.truth.fcsf > 0.3) = 3;
cls = {'WM-like','GM-like','CSF-like'};
base = struct('solver','mcmc','algorithm','MH','metric',{{'median','std'}},'iteration',20000,'burnin',10000,'thinning',1, ...
              'repetition',4,'start','likelihood','mcmcClass','mcmc_bayes','likelihood','marginal_noise', ...
              'parameterTransform','sigmoid','updateScheme','joint','adaptStepSize',true,'adaptCovariance',true,'overdisp',0.01);
if isPilot; base.iteration = 1000; base.burnin = 500; end

fprintf('T9b.2 Student-t prior on the ME-AxCaliberSMT phantom (characterisation, no pass/fail)%s\n', repmat(' [PILOT]', 1, isPilot));
fprintf('%s, %d voxels (WM/GM/CSF-like %s); iteration %d, burn-in %d, 4 repetitions; seed 21 per arm\n\n', ...
    mat2str(size(m)), nnz(m), mat2str(arrayfun(@(c) nnz(P.label == c), 1:3)), base.iteration, base.burnin);

arms = {'D','D_K2','D_t'};
R = struct();
for a = 1:numel(arms)
    f = base;
    switch arms{a}
        case 'D';    f.prior = struct('hierarchical', struct('params', {pn}));
        case 'D_K2'; f.prior = struct('hierarchical', struct('params', {pn}, 'K', 2));
        case 'D_t';  f.prior = struct('hierarchical', struct('params', {pn}, 'distribution', 't', 'nu', 4));
    end
    rng(21); parallel.gpu.rng(21);
    tt = tic; [~, out] = evalc('P.obj().estimate(P.data, P.mask, f, []);'); S.t = toc(tt);
    for k = 1:numel(pn)
        xk = out.posterior.(pn{k}); xk = reshape(xk, size(xk,1), [], f.repetition);
        S.(pn{k}).median = out.median.(pn{k});
        S.(pn{k}).q5  = zeros(size(m)); S.(pn{k}).q5(idx)  = sample_quantile(reshape(xk, size(xk,1), []), 0.05, 2);
        S.(pn{k}).q95 = zeros(size(m)); S.(pn{k}).q95(idx) = sample_quantile(reshape(xk, size(xk,1), []), 0.95, 2);
        S.(pn{k}).rhat = mcmc_bayes.rhat(xk);
    end
    S.hyper = out.hyper;
    R.(arms{a}) = S; clear S out xk
    fprintf('%s done (%.1f min)\n', arms{a}, R.(arms{a}).t/60);
end

% ---- report ----
for k = [4 5 1 2 3]
    t = P.truth.(pn{k});
    fprintf('\n=== %s ===\n%-6s', pn{k}, 'arm');
    for c = 1:3; fprintf(' | %-22s', sprintf('%s bias RMSE cov', cls{c})); end
    fprintf(' | all RMSE cov | voxel R-hat med, >1.01\n');
    for a = 1:numel(arms)
        S = R.(arms{a}).(pn{k}); fprintf('%-6s', arms{a});
        for c = 1:3
            mm = m & P.label == c; e = S.median(mm) - t(mm); cv = mean(t(mm) >= S.q5(mm) & t(mm) <= S.q95(mm));
            fprintf(' | %+7.3f %6.3f %4.2f    ', mean(e), sqrt(mean(e.^2)), cv);
        end
        e = S.median(m) - t(m); cv = mean(t(m) >= S.q5(m) & t(m) <= S.q95(m));
        fprintf(' | %6.3f %4.2f | %.3f %.3f\n', sqrt(mean(e.^2)), cv, median(S.rhat), mean(S.rhat > 1.01));
    end
end
fprintf('\nhyperparameter R-hat (max over entries) and run time:\n');
for a = 1:numel(arms)
    H = R.(arms{a}).hyper;
    fprintf('%-6s', arms{a});
    if isfield(H, 'rhat')
        fprintf(' mu %.3f, Sigma %.3f', max(H.rhat.mu(:)), max(H.rhat.Sigma(:)));
        if isfield(H.rhat, 'pi'); fprintf(', pi %.3f', max(H.rhat.pi(:))); end
    end
    fprintf(' | %.1f min\n', R.(arms{a}).t/60);
end
lam = R.D_t.hyper.lambda;
fprintf('\nD_t posterior mean lambda (prior mean 1; small = treated as an outlier):\n');
for c = 1:3
    mm = m & P.label == c;
    fprintf('  %-9s median %.3f, 10th pct %.3f\n', cls{c}, median(lam(mm)), sample_quantile(lam(mm), 0.10, 1));
end
fprintf('\nT9b.2 total time %.1f min\n', toc(tStart)/60);

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
