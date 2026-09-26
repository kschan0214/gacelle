%% test_T3_2_sbc_linear_gaussian.m
%
% T3.2 (Phase 3): simulation-based calibration (Talts et al. 2018,
% arXiv:1804.06788) of the hierarchical Normal prior with FREE NIW
% hyperparameters on the linear Gaussian toy.
%
% Generative model (per replicate r = 1..R), d = 2, m = 6, known s = 1:
%   Sigma ~ IW(Psi0, nu0), mu | Sigma ~ N(m0, Sigma/kappa0),
%   u_i ~ N(mu, Sigma) (i = 1..Nv), y_i = A u_i + e_i, e_i ~ N(0, s^2 I).
%   NIW(m0 = [0 0], kappa0 = 1, Psi0 = 2 I, nu0 = 5) (E[Sigma] = I). A is fixed
%   (randn, seeded). Sigma is generated independently of the sampler's Bartlett
%   code: Sigma^-1 = sum_{k=1..nu0} z_k z_k', z_k ~ N(0, Psi0^-1) (integer nu0).
% Inference: mcmc_bayes, likelihood 'gaussian' with known noise (test-only
%   fitting.fixedParams), u1, u2 'linear' with lb = -Inf, ub = Inf, prior.hierarchical
%   with the same NIW hyperprior (free mode, Gibbs block after every MH sweep),
%   joint updates, adaptStepSize (burn-in only), one chain (repetition = 1),
%   start u_i = least-squares estimate A\y_i (data only).
%
% Rank statistics. Quantities: mu1, mu2, Sigma11, Sigma21, Sigma22 and, for 3
%   voxels drawn at random in each replicate (slots v1..v3), u1 and u2: K = 11.
%   For each quantity the chain (Ns retained draws) is thinned to near
%   independence with stride ceil(Ns/ESS) (ESS: mcmc_bayes.ess, split chain),
%   and the last L = 99 thinned draws are used: rank = #{draws < truth} in 0..L.
%   If Ns/stride < L the stride is reduced to floor(Ns/L) (residual
%   autocorrelation, conservative: it can only widen the rank histogram's
%   deviations); the number of such cases is reported.
% Criterion (stated before running): for each quantity, chi-square test of
%   uniformity over 10 bins of 10 ranks (df = 9, p from gammainc). PASS if all
%   K = 11 p-values are >= 0.05/K (Bonferroni, family-wise alpha = 0.05).
%   Also reported (information only): the pooled voxel ranks (3 slots x 2
%   parameters x R), the smallest p-value, and the rank histograms with the 99%
%   binomial band of each bin (PNG in outDir).
%
% Size: R = 200 replicates x Nv = 500 voxels (the draft's sizes).
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T3_2_sbc_linear_gaussian
% The PNG and a .mat with the ranks go to getenv('MCMC_BAYES_OUTDIR'), or tempdir.
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%

clearvars; tStart = tic;

% settings (R and Nv can be overridden for a pilot with the environment
% variables MCMC_BAYES_SBC_R / MCMC_BAYES_SBC_NV)
R   = 200; Nv = 500;
if ~isempty(getenv('MCMC_BAYES_SBC_R'));  R  = str2double(getenv('MCMC_BAYES_SBC_R'));  end
if ~isempty(getenv('MCMC_BAYES_SBC_NV')); Nv = str2double(getenv('MCMC_BAYES_SBC_NV')); end
d = 2; m = 6; s = 1; L = 99; nBins = 10; Nslot = 3;
m0 = [0; 0]; kappa0 = 1; nu0 = 5; Psi0 = 2*eye(d);
iteration = 15000; burnin = 3000; thinning = 3;
alphaFW = 0.05;
seedA = 81; seedBase = 1000;
outDir = getenv('MCMC_BAYES_OUTDIR'); if isempty(outDir); outDir = tempdir; end

rng(seedA); A = randn(m, d);
names = {'mu1','mu2','Sigma11','Sigma21','Sigma22'};
for k = 1:Nslot; names = [names, {sprintf('v%d.u1',k), sprintf('v%d.u2',k)}]; end %#ok<AGROW>
K = numel(names);
fprintf('T3.2 SBC linear Gaussian, free NIW hyperparameters: R=%d replicates x Nv=%d voxels, d=%d, m=%d, s=%g\n', R, Nv, d, m, s);
fprintf('NIW(m0=%s, kappa0=%g, Psi0=%s, nu0=%g); sampler %d iterations, burn-in %d, thinning %d\n', ...
    mat2str(m0.'), kappa0, mat2str(Psi0), nu0, iteration, burnin, thinning);
fprintf('Seeds: A %d, replicate r uses rng/gpu rng %d + r\n', seedA, seedBase);
fprintf('Criterion: chi-square (10 bins, L=%d) p >= %.4f for all K=%d quantities (Bonferroni, alpha %.2f)\n\n', L, alphaFW/K, K, alphaFW);

f.modelParams = {'u1';'u2';'noise'};
f.lb = [-Inf; -Inf; 0]; f.ub = [Inf; Inf; 10]; f.xStepSize = [0.3; 0.3; 0.01];
f.algorithm = 'MH'; f.iteration = iteration; f.burnin = burnin; f.thinning = thinning; f.metric = {'mean'};
f.repetition = 1; f.fixedParams = struct('noise', s);
f.adaptStepSize = true; f.adaptInterval = 50;
f.prior.hierarchical = struct('hyperprior','niw','m0',m0,'kappa0',kappa0,'Psi0',Psi0,'nu0',nu0);
mask = true(Nv, 1);
fwd  = @(p) lingauss_fwd(p, A);

ranks = zeros(R, K); essQ = zeros(R, K); reduced = false(R, K); tRep = zeros(R, 1);
for r = 1:R
    rng(seedBase + r); parallel.gpu.rng(seedBase + r);
    % truth from the hyperprior (Wishart by outer products, independent of the sampler's Bartlett code)
    Z       = chol(inv(Psi0), 'lower') * randn(d, nu0);
    Sigma   = inv(Z*Z.'); Sigma = (Sigma + Sigma.')/2;
    mu      = m0 + chol(Sigma/kappa0, 'lower')*randn(d, 1);
    u       = mu + chol(Sigma, 'lower')*randn(d, Nv);
    y       = A*u + s*randn(m, Nv);
    slots   = randperm(Nv, Nslot);
    uLS     = A \ y;
    x0      = struct('u1', uLS(1,:).', 'u2', uLS(2,:).', 'noise', s*ones(Nv,1));
    yy      = reshape(y.', [Nv 1 1 m]);

    t0 = tic;
    evalc('out = mcmc_bayes().optimisation(yy, mask, [], x0, f, fwd);');
    tRep(r) = toc(t0);

    H       = out.hyper.posterior;
    draws   = {H.mu(1,:), H.mu(2,:), squeeze(H.Sigma(1,1,:)).', squeeze(H.Sigma(2,1,:)).', squeeze(H.Sigma(2,2,:)).'};
    truth   = [mu(1), mu(2), Sigma(1,1), Sigma(2,1), Sigma(2,2)];
    for k = 1:Nslot
        draws = [draws, {out.posterior.u1(slots(k),:), out.posterior.u2(slots(k),:)}]; %#ok<AGROW>
        truth = [truth, u(1,slots(k)), u(2,slots(k))]; %#ok<AGROW>
    end
    for k = 1:K
        [ranks(r,k), essQ(r,k), reduced(r,k)] = sbc_rank(double(draws{k}), truth(k), L);
    end
    if r == 1 || mod(r, 20) == 0
        fprintf('replicate %3d/%d: %.1f s, ESS mu %s Sigma %s, elapsed %.1f min\n', r, R, tRep(r), ...
            mat2str(essQ(r,1:2),3), mat2str(essQ(r,3:5),3), toc(tStart)/60);
    end
end

% chi-square uniformity tests
[p, X2, counts] = chi2_uniform(ranks, L, nBins);
pPool = chi2_uniform(reshape(ranks(:, 6:end), [], 1), L, nBins);
pMin  = alphaFW / K;
isPass = all(p >= pMin);
fprintf('\n%-10s %8s %8s %10s %9s\n', 'quantity', 'X2', 'p', 'ESS med', '#reduced');
for k = 1:K
    fprintf('%-10s %8.2f %8.4f %10.0f %9d%s\n', names{k}, X2(k), p(k), median(essQ(:,k)), nnz(reduced(:,k)), flag(p(k) < pMin));
end
fprintf('pooled voxel ranks (%d, information only): p = %.4f\n', numel(ranks(:,6:end)), pPool);
fprintf('min p = %.4f (threshold %.4f); runtime per replicate median %.1f s, total %.1f min\n', min(p), pMin, median(tRep), toc(tStart)/60);

% rank histograms with the 99% binomial band per bin
[lo, hi] = binom_bounds(R, 1/nBins, 0.01);
fig = figure('Visible','off','Position',[0 0 1400 700]);
for k = 1:K
    subplot(3, 4, k); hold on;
    patch([0.5 nBins+0.5 nBins+0.5 0.5], [lo lo hi hi], [0.85 0.85 0.85], 'EdgeColor','none');
    bar(1:nBins, counts(:,k), 1, 'FaceColor', [0.3 0.45 0.7]);
    yline(R/nBins, 'k--');
    title(sprintf('%s  p=%.3f', names{k}, p(k)), 'Interpreter','none'); xlim([0.5 nBins+0.5]);
end
sgtitle(sprintf('T3.2 SBC linear Gaussian (R=%d, Nv=%d, L=%d): rank histograms, grey = 99%% binomial band', R, Nv, L));
pngFile = fullfile(outDir, 'T3_2_sbc_rank_histograms.png');
exportgraphics(fig, pngFile); close(fig);
save(fullfile(outDir, 'T3_2_sbc_ranks.mat'), 'ranks', 'essQ', 'reduced', 'names', 'p', 'X2', 'counts', 'tRep', 'R', 'Nv', 'L');
fprintf('saved %s\n', pngFile);

fprintf('\nT3.2 overall: %s   (total time %.1f min)\n', pf(isPass), toc(tStart)/60);

%% local functions
function [rk, e, isReduced] = sbc_rank(x, truth, L)
% rank of the truth among L draws thinned to near independence (see header)
Ns      = numel(x);
e       = mcmc_bayes.ess(reshape(x, 1, Ns, 1));
stride  = max(1, ceil(Ns/e));
isReduced = floor(Ns/stride) < L;
if isReduced; stride = floor(Ns/L); end
idx     = Ns - (L-1:-1:0)*stride;
rk      = sum(x(idx) < truth);
end

function [p, X2, counts] = chi2_uniform(ranks, L, nBins)
% chi-square test of uniform ranks 0..L over nBins equal bins, per column
edges   = linspace(-0.5, L+0.5, nBins+1);
counts  = zeros(nBins, size(ranks,2));
for k = 1:size(ranks,2); counts(:,k) = histcounts(ranks(:,k), edges).'; end
E       = size(ranks,1)/nBins;
X2      = sum((counts - E).^2 ./ E, 1);
p       = gammainc(X2/2, (nBins-1)/2, 'upper');
end

function s = pf(c)
if c; s = 'PASS'; else; s = 'FAIL'; end
end

function s = flag(c)
if c; s = '  <-- below threshold'; else; s = ''; end
end
