%% test_T3_3_sbc_ivim.m
%
% T3.3 (Phase 3): simulation-based calibration of the hierarchical Normal
% prior (free NIW hyperparameters) on small IVIM datasets with the
% 'marginal_S0noise' likelihood (Zellner broad limit on S0, 1/sigma^2).
%
% Parameters and transforms (u = T(x), all three under the hierarchy, d = 3):
%   D     'log'     (lb 0, ub Inf)   u1 = log D        [um^2/ms]
%   F     'sigmoid' (lb 0, ub 1)     u2 = logit F
%   Dstar 'log'     (lb 0, ub Inf)   u3 = log Dstar    [um^2/ms]
%   S0 and sigma are marginalised. b-values of ivim_t2_config.m (0..900 s/mm^2).
%
% Hyperprior (generating = inference), chosen to give plausible IVIM values:
%   m0 = [log 1, logit 0.1, log 20], kappa0 = 1, nu0 = 8,
%   Psi0 = (nu0-d-1) diag([0.2 0.4 0.3].^2)  ->  E[Sigma] = diag(0.04, 0.16, 0.09).
%   Prior predictive: mu +- SD(mu) and 95% of voxels around mu roughly give
%   D in [0.5, 2], F in [0.03, 0.3], Dstar in [8, 50] um^2/ms.
%   Label switching (D <-> Dstar, F <-> 1-F) is suppressed by the prior on F
%   (the swapped mode is > 10 prior SD away).
%
% Nuisance generation. The broad-limit Zellner prior is improper, so exact SBC
%   needs a generator that matches it locally:
%   * sigma is a scale parameter with the right-Haar prior 1/sigma: the posterior
%     of (u, S0/sigma) depends on y only through y/|y|, so any sigma distribution
%     is exact.
%   * S0 | sigma, u ~ N(0, k sigma^2/c(u)), k -> Inf, c = g'g: locally flat in the
%     'energy SNR' rho = S0 sqrt(c(u))/sigma, NOT in S0/sigma (which would tilt the
%     u-posterior by sqrt(c(u))). So rho ~ U(50, 125), independent of u, and
%     sigma_i = S0_i |g_i| / rho_i, S0_i ~ U(0.8, 1.2) (irrelevant by invariance).
%     b = 0 SNR S0/sigma = rho/|g| is ~16-40. The only mismatch is the edge of the
%     uniform rho (posterior SD of rho ~1 on a width of 75): negligible.
%
% Sampler: joint updates, adaptStepSize (burn-in only), one chain, start at
%   the prior centre D = 1, F = 0.1, Dstar = 20 for all voxels (data-independent).
%   24000 iterations, burn-in 6000, thinning 6 (3000 draws). A pilot (3 replicates,
%   20000 iterations) gave ESS of ~30-110 (of 3000 draws) for mu_Dstar and the
%   Sigma entries involving Dstar (the centred hierarchy mixes slowly where the
%   data inform Dstar weakly), hence the longer chains and L = 29.
%
% Rank statistics as in T3.2: quantities mu (3), Sigma lower triangle (6) and,
%   for 2 random voxels per replicate, u1..u3 (6): K = 15. Thinning to near
%   independence by ESS, L = 29 draws (ranks 0..29), 10 bins of 3.
% Size: R = 100 replicates x Nv = 150 voxels (reduced from the draft's 200 x 500
%   on runtime grounds, see the report).
% Criterion (stated before running): chi-square uniformity (10 bins, df 9) for
%   each quantity; PASS if all K = 15 p-values >= 0.05/K (Bonferroni).
%   Power (information): with R = 100, a chi-square test over 10 bins detects
%   only gross miscalibration (e.g. a posterior SD off by ~30%, or a mean bias of
%   ~0.3 posterior SD); smaller errors can pass.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T3_3_sbc_ivim
% The PNG and a .mat with the ranks go to getenv('MCMC_BAYES_OUTDIR'), or tempdir.
% MCMC_BAYES_SBC_R / MCMC_BAYES_SBC_NV override R / Nv (pilot runs).
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%

clearvars; tStart = tic;

R   = 100; Nv = 150;
if ~isempty(getenv('MCMC_BAYES_SBC_R'));  R  = str2double(getenv('MCMC_BAYES_SBC_R'));  end
if ~isempty(getenv('MCMC_BAYES_SBC_NV')); Nv = str2double(getenv('MCMC_BAYES_SBC_NV')); end
d = 3; L = 29; nBins = 10; Nslot = 2;
m0      = [log(1); log(0.1/0.9); log(20)];
kappa0  = 1; nu0 = 8; Psi0 = (nu0-d-1)*diag([0.2 0.4 0.3].^2);
rhoRange = [50 125]; S0Range = [0.8 1.2];
iteration = 24000; burnin = 6000; thinning = 6;
alphaFW = 0.05;
seedBase = 2000;
outDir = getenv('MCMC_BAYES_OUTDIR'); if isempty(outDir); outDir = tempdir; end

cfg = ivim_t2_config();
b   = cfg.b; m = numel(b);
names = {'mu_D','mu_F','mu_Ds','Sig_DD','Sig_FD','Sig_DsD','Sig_FF','Sig_DsF','Sig_DsDs'};
for k = 1:Nslot; names = [names, {sprintf('v%d.D',k), sprintf('v%d.F',k), sprintf('v%d.Ds',k)}]; end %#ok<AGROW>
K = numel(names);
iLow = find(tril(true(d)));                 % Sigma lower triangle, column-major
fprintf('T3.3 SBC IVIM, marginal_S0noise, free NIW: R=%d x Nv=%d, %d b-values\n', R, Nv, m);
fprintf('NIW(m0=%s, kappa0=%g, nu0=%g, Psi0=diag %s); rho ~ U%s, S0 ~ U%s\n', mat2str(m0.',3), kappa0, nu0, ...
    mat2str(diag(Psi0).',3), mat2str(rhoRange), mat2str(S0Range));
fprintf('Sampler %d iterations, burn-in %d, thinning %d; seeds %d + r\n', iteration, burnin, thinning, seedBase);
fprintf('Criterion: chi-square (10 bins, L=%d) p >= %.4f for all K=%d quantities (Bonferroni, alpha %.2f)\n\n', L, alphaFW/K, K, alphaFW);

f.modelParams   = {'S0';'D';'F';'Dstar';'noise'};
f.lb            = [0; 0;   0; 0;   0.001];
f.ub            = [2; Inf; 1; Inf; 1];
f.xStepSize     = [0.01; 0.02; 0.01; 2; 0.001];
f.parameterTransform = {'linear','log','sigmoid','log','linear'};
f.likelihood    = 'marginal_S0noise'; f.S0Param = 'S0';
f.algorithm = 'MH'; f.iteration = iteration; f.burnin = burnin; f.thinning = thinning; f.metric = {'mean'};
f.repetition = 1; f.adaptStepSize = true; f.adaptInterval = 50;
f.prior.hierarchical = struct('hyperprior','niw','m0',m0,'kappa0',kappa0,'Psi0',Psi0,'nu0',nu0);
mask = true(Nv, 1);
fwd  = @(p) ivim_fwd(p, b);
x0   = struct('S0', ones(Nv,1), 'D', ones(Nv,1), 'F', 0.1*ones(Nv,1), 'Dstar', 20*ones(Nv,1), 'noise', 0.02*ones(Nv,1));

ranks = zeros(R, K); essQ = zeros(R, K); reduced = false(R, K); tRep = zeros(R, 1);
xRange = zeros(R, 6);
for r = 1:R
    rng(seedBase + r); parallel.gpu.rng(seedBase + r);
    Z       = chol(inv(Psi0), 'lower') * randn(d, nu0);
    Sigma   = inv(Z*Z.'); Sigma = (Sigma + Sigma.')/2;
    mu      = m0 + chol(Sigma/kappa0, 'lower')*randn(d, 1);
    u       = mu + chol(Sigma, 'lower')*randn(d, Nv);
    p.D     = exp(u(1,:)); p.F = 1./(1+exp(-u(2,:))); p.Dstar = exp(u(3,:));
    g       = ivim_fwd(p, b);                                   % [m, Nv], no S0
    S0      = S0Range(1) + diff(S0Range)*rand(1, Nv);
    rho     = rhoRange(1) + diff(rhoRange)*rand(1, Nv);
    sigma   = S0 .* sqrt(sum(g.^2, 1)) ./ rho;
    y       = S0.*g + sigma.*randn(m, Nv);
    slots   = randperm(Nv, Nslot);
    yy      = reshape(y.', [Nv 1 1 m]);
    xRange(r,:) = [min(p.D) max(p.D) min(p.F) max(p.F) min(p.Dstar) max(p.Dstar)];

    t0 = tic;
    evalc('out = mcmc_bayes().optimisation(yy, mask, [], x0, f, fwd);');
    tRep(r) = toc(t0);

    H       = out.hyper.posterior;
    SigF    = reshape(H.Sigma, d*d, []);
    draws   = [num2cell(H.mu, 2).', num2cell(SigF(iLow,:), 2).'];
    truth   = [mu.', Sigma(iLow).'];
    for k = 1:Nslot
        v = slots(k);
        draws = [draws, {log(out.posterior.D(v,:)), log(out.posterior.F(v,:)./(1-out.posterior.F(v,:))), log(out.posterior.Dstar(v,:))}]; %#ok<AGROW>
        truth = [truth, u(:,v).']; %#ok<AGROW>
    end
    for k = 1:K
        [ranks(r,k), essQ(r,k), reduced(r,k)] = sbc_rank(double(draws{k}), truth(k), L);
    end
    if r == 1 || mod(r, 10) == 0
        fprintf('replicate %3d/%d: %.1f s, ESS min over mu/Sigma %.0f, voxel ESS min %.0f, elapsed %.1f min\n', r, R, tRep(r), ...
            min(essQ(r,1:9)), min(essQ(r,10:end)), toc(tStart)/60);
    end
end

[p, X2, counts] = chi2_uniform(ranks, L, nBins);
pPool = chi2_uniform(reshape(ranks(:, 10:end), [], 1), L, nBins);
pMin  = alphaFW / K;
isPass = all(p >= pMin);
fprintf('\nsimulated ranges (median over replicates of min/max): D [%.2f %.2f], F [%.3f %.3f], Dstar [%.1f %.1f]\n', median(xRange,1));
fprintf('%-10s %8s %8s %10s %9s\n', 'quantity', 'X2', 'p', 'ESS med', '#reduced');
for k = 1:K
    fprintf('%-10s %8.2f %8.4f %10.0f %9d%s\n', names{k}, X2(k), p(k), median(essQ(:,k)), nnz(reduced(:,k)), flag(p(k) < pMin));
end
fprintf('pooled voxel ranks (%d, information only): p = %.4f\n', numel(ranks(:,10:end)), pPool);
fprintf('min p = %.4f (threshold %.4f); runtime per replicate median %.1f s, total %.1f min\n', min(p), pMin, median(tRep), toc(tStart)/60);

[lo, hi] = binom_bounds(R, 1/nBins, 0.01);
fig = figure('Visible','off','Position',[0 0 1500 800]);
for k = 1:K
    subplot(3, 5, k); hold on;
    patch([0.5 nBins+0.5 nBins+0.5 0.5], [lo lo hi hi], [0.85 0.85 0.85], 'EdgeColor','none');
    bar(1:nBins, counts(:,k), 1, 'FaceColor', [0.3 0.45 0.7]);
    yline(R/nBins, 'k--');
    title(sprintf('%s  p=%.3f', names{k}, p(k)), 'Interpreter','none'); xlim([0.5 nBins+0.5]);
end
sgtitle(sprintf('T3.3 SBC IVIM (R=%d, Nv=%d, L=%d): rank histograms, grey = 99%% binomial band', R, Nv, L));
pngFile = fullfile(outDir, 'T3_3_sbc_rank_histograms.png');
exportgraphics(fig, pngFile); close(fig);
save(fullfile(outDir, 'T3_3_sbc_ranks.mat'), 'ranks', 'essQ', 'reduced', 'names', 'p', 'X2', 'counts', 'tRep', 'R', 'Nv', 'L', 'xRange');
fprintf('saved %s\n', pngFile);

fprintf('\nT3.3 overall: %s   (total time %.1f min)\n', pf(isPass), toc(tStart)/60);

%% local functions
function [rk, e, isReduced] = sbc_rank(x, truth, L)
% rank of the truth among L draws thinned to near independence (as in T3.2)
Ns      = numel(x);
e       = mcmc_bayes.ess(reshape(x, 1, Ns, 1));
stride  = max(1, ceil(Ns/e));
isReduced = floor(Ns/stride) < L;
if isReduced; stride = floor(Ns/L); end
idx     = Ns - (L-1:-1:0)*stride;
rk      = sum(x(idx) < truth);
end

function [p, X2, counts] = chi2_uniform(ranks, L, nBins)
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
