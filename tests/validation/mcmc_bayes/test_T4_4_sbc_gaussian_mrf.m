%% test_T4_4_sbc_gaussian_mrf.m
%
% T4.4 (Phase 4): simulation-based calibration (Talts et al. 2018) of the chromatic MRF sampler
% with a Gaussian MRF, fixed mu, Sigma and tau.
%
% Generative model per replicate (6 x 6 x 3 grid, mask with ~15% holes, Nv voxels, d = 2):
%   u ~ N(1 (x) mu, P0^-1),  P0 = I (x) Sigma^-1 + (1/tau) L (x) diag(W)
%   (the hierarchical prior times exp(-Phi_MRF) with rho = x^2/2 and w_ij = 1; its mean is
%   1 (x) mu because L 1 = 0), drawn exactly by sparse Cholesky: P0 = R'R, u = 1 (x) mu + R \ z.
%   L is built by brute-force enumeration of the masked pairs. y_i = A u_i + e_i, known s = 1,
%   m = 4. mu = [0.5 -1], SDs [0.8 0.5], corr 0.3, W = 1./sqrt(diag(Sigma)) (the default).
% Inference: mcmc_bayes with the same prior (hierarchical fixed + mrf quadratic), joint updates,
%   adaptStepSize (burn-in only), one chain per replicate, start u = mu (data independent),
%   12000 iterations, burn-in 2000, thinning 5 (2000 draws).
%   All R replicates run in ONE call: R copies of the grid stacked along dim 3, separated by an
%   empty slice (no edges between replicates; they share mu, Sigma, tau), each with its own u, y.
% Modes: '3d' face with tau = 0.5, '2d' full r = 2 with tau = 1.5 (the "strong" couplings of T4.2).
%
% Rank statistics, K = 10 quantities per replicate: u1, u2 of 3 random voxels (slots v1..v3),
%   the replicate mean of u1 and of u2 over all voxels, and u1_i - u1_j, u2_i - u2_j across one
%   random edge (functions of u are valid SBC quantities; the last two are sensitive to the
%   neighbour correlations). As in T3.2: thin to near independence with stride ceil(Ns/ESS) and
%   use the last L = 99 thinned draws (rank 0..99); if Ns/stride < L the stride is reduced to
%   floor(Ns/L) (reported).
% Criterion (stated before running): chi-square uniformity over 10 bins (df 9) per quantity and
%   mode; PASS if all 2 x 10 p-values >= 0.05/20 (Bonferroni, family-wise 0.05).
% Size: R = 1000 replicates per mode (one GPU call each).
% Power (information, printed): the chance that one quantity's chi-square test at 0.05/20 rejects,
%   by Monte Carlo on the host for a Gaussian posterior whose SD is off by a factor (1 +/- e) or
%   whose mean is off by b posterior SDs.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T4_4_sbc_gaussian_mrf
% MCMC_BAYES_SBC_R overrides R (pilot); MCMC_BAYES_SBC_MODE / MCMC_BAYES_SBC_SEEDOFFSET select one
% mode and shift its seed (follow-up runs; the Bonferroni threshold then covers that mode only). PNG and .mat to getenv('MCMC_BAYES_OUTDIR') or tempdir.
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%

clearvars; tStart = tic;

R = 1000;
if ~isempty(getenv('MCMC_BAYES_SBC_R')); R = str2double(getenv('MCMC_BAYES_SBC_R')); end
d = 2; m = 4; s = 1; dims = [6 6 3]; L = 99; nBins = 10; Nslot = 3;
mu = [0.5; -1]; sds = [0.8; 0.5]; Sigma = diag(sds)*[1 0.3; 0.3 1]*diag(sds); W = 1./sqrt(diag(Sigma));
iteration = 12000; burnin = 2000; thinning = 5;
modes = { {'3d', 1, 'face', 0.5}, {'2d', 2, 'full', 1.5} };
alphaFW = 0.05; Ktot = 10*numel(modes); pMin = alphaFW/Ktot;
outDir = getenv('MCMC_BAYES_OUTDIR'); if isempty(outDir); outDir = tempdir; end
seedGeom = 201; seedA = 202; seedMode = [210 220];
% follow-up runs: MCMC_BAYES_SBC_MODE = '3d' | '2d' selects one mode, MCMC_BAYES_SBC_SEEDOFFSET shifts
% the per-mode seeds (independent replicates; geometry and A unchanged)
if ~isempty(getenv('MCMC_BAYES_SBC_SEEDOFFSET')); seedMode = seedMode + str2double(getenv('MCMC_BAYES_SBC_SEEDOFFSET')); end
if ~isempty(getenv('MCMC_BAYES_SBC_MODE'))
    keep = cellfun(@(c) strcmp(c{1}, getenv('MCMC_BAYES_SBC_MODE')), modes);
    modes = modes(keep); seedMode = seedMode(keep); Ktot = 10*numel(modes); pMin = alphaFW/Ktot;
end

rng(seedGeom); mask1 = rand(dims) > 0.15; idx1 = find(mask1); Nv = numel(idx1);
rng(seedA); A = randn(m, d);
names = [arrayfun(@(k) sprintf('v%d.u1',k), 1:Nslot, 'UniformOutput', false), ...
         arrayfun(@(k) sprintf('v%d.u2',k), 1:Nslot, 'UniformOutput', false), ...
         {'mean.u1','mean.u2','edge.du1','edge.du2'}];
K = numel(names);
fprintf('T4.4 SBC Gaussian MRF: R=%d replicates x Nv=%d voxels (grid %s), d=%d, m=%d, s=%g\n', R, Nv, mat2str(dims), d, m, s);
fprintf('sampler %d iterations, burn-in %d, thinning %d; seeds geometry %d, A %d, modes %s (rng and gpu rng)\n', ...
    iteration, burnin, thinning, seedGeom, seedA, mat2str(seedMode));
fprintf('Criterion: chi-square (10 bins, L=%d) p >= %.5f for all %d quantities x %d modes (Bonferroni, alpha %.2f)\n\n', L, pMin, K, numel(modes), alphaFW);

% power (information): P(reject at pMin) for one quantity, R replicates
rng(230);
powE = [0.05 0.10 0.20]; powB = [0.1 0.2 0.3]; Npow = 400;
pw = zeros(1, numel(powE) + numel(powB));
for kk = 1:numel(pw)
    rej = 0;
    for t = 1:Npow
        th = randn(R, 1);                                   % truth relative to the true posterior
        if kk <= numel(powE); dr = (1 + powE(kk)) * randn(R, L); else; dr = powB(kk-numel(powE)) + randn(R, L); end
        rej = rej + (chi2_uniform(sum(dr < th, 2), L, nBins) < pMin);
    end
    pw(kk) = rej / Npow;
end
fprintf('power per quantity (R=%d): SD x1.05 %.2f, x1.10 %.2f, x1.20 %.2f | mean bias 0.1 SD %.2f, 0.2 SD %.2f, 0.3 SD %.2f\n\n', R, pw);

isPass = false(1, numel(modes)); allP = [];
for km = 1:numel(modes)
    [mode, r, conn, tau] = modes{km}{:};
    rng(seedMode(km)); parallel.gpu.rng(seedMode(km));
    % brute-force Laplacian and one edge list of the replicate grid
    [i, j, k] = ind2sub(dims, idx1); P = [i j k];
    D = abs(permute(P, [1 3 2]) - permute(P, [3 1 2]));
    if strcmp(conn, 'face'); Adj = sum(D, 3) == 1; else; Adj = max(D, [], 3) <= r & max(D, [], 3) > 0; end
    if strcmp(mode, '2d'); Adj = Adj & D(:,:,3) == 0; end
    Lap = diag(sum(Adj, 2)) - double(Adj);
    [ea, eb] = find(triu(Adj));
    P0  = kron(speye(Nv), sparse(inv(Sigma))) + kron(sparse(Lap), sparse(diag(W)))./tau;
    Rc  = chol(P0);                                         % P0 = Rc'Rc
    % replicates stacked along dim 3 with one empty slice between them
    Z   = R*dims(3) + (R-1);
    mask = false([dims(1:2) Z]); yy = zeros([dims(1:2) Z m], 'single');
    uTrue = zeros(d, Nv, R);
    for rr = 1:R
        u  = reshape(repmat(mu, Nv, 1) + Rc \ randn(Nv*d, 1), d, Nv);
        y  = A*u + s*randn(m, Nv);
        uTrue(:,:,rr) = u;
        zs = (rr-1)*(dims(3)+1) + (1:dims(3));
        mask(:,:,zs) = mask1;
        yi = zeros(numel(mask1), m); yi(idx1, :) = y.';
        yy(:,:,zs,:) = reshape(yi, [dims m]);
    end
    slots = zeros(R, Nslot); edge = zeros(R, 1);
    for rr = 1:R; slots(rr,:) = randperm(Nv, Nslot); edge(rr) = randi(numel(ea)); end

    f = struct();
    f.modelParams = {'u1';'u2';'noise'}; f.lb = [-Inf; -Inf; 0]; f.ub = [Inf; Inf; 10]; f.xStepSize = [0.3; 0.3; 0.01];
    f.algorithm = 'MH'; f.iteration = iteration; f.burnin = burnin; f.thinning = thinning; f.metric = {'mean'};
    f.repetition = 1; f.fixedParams = struct('noise', s);
    f.adaptStepSize = true; f.adaptInterval = 50;
    f.prior.hierarchical = struct('fixed', true, 'mu', mu, 'Sigma', Sigma);
    f.prior.mrf = struct('potential', 'quadratic', 'tau', tau, 'mode', mode, 'radius', r, 'connectivity', conn);
    x0 = struct('u1', mu(1)*ones(size(mask)), 'u2', mu(2)*ones(size(mask)));
    t0 = tic;
    evalc('out = mcmc_bayes().optimisation(yy, mask, [], x0, f, @(pp) lingauss_fwd(pp, A));');
    tRun = toc(t0);
    U1 = reshape(out.posterior.u1, Nv, R, []); U2 = reshape(out.posterior.u2, Nv, R, []);   % voxel order within replicate
    Ns = size(U1, 3);
    clear out

    ranks = zeros(R, K); essQ = zeros(R, K); reduced = false(R, K);
    for rr = 1:R
        u1 = double(squeeze(U1(:,rr,:))); u2 = double(squeeze(U2(:,rr,:)));     % [Nv, Ns]
        sl = slots(rr,:); ia = ea(edge(rr)); ib = eb(edge(rr)); ut = uTrue(:,:,rr);
        draws = [num2cell(u1(sl,:), 2).', num2cell(u2(sl,:), 2).', {mean(u1,1), mean(u2,1), u1(ia,:)-u1(ib,:), u2(ia,:)-u2(ib,:)}];
        truth = [ut(1,sl), ut(2,sl), mean(ut(1,:)), mean(ut(2,:)), ut(1,ia)-ut(1,ib), ut(2,ia)-ut(2,ib)];
        for kq = 1:K
            [ranks(rr,kq), essQ(rr,kq), reduced(rr,kq)] = sbc_rank(draws{kq}, truth(kq), L);
        end
    end
    [p, X2, counts] = chi2_uniform(ranks, L, nBins);
    allP = [allP p]; %#ok<AGROW>
    isPass(km) = all(p >= pMin);
    fprintf('mode %s r=%d %s tau=%g: Nv total %d, colours %d, run %.1f min, %d draws per replicate\n', mode, r, conn, tau, Nv*R, ...
        numel(unique(mcmc_bayes.build_colours(idx1, dims, mode, r, conn))), tRun/60, Ns);
    fprintf('%-10s %8s %8s %10s %9s\n', 'quantity', 'X2', 'p', 'ESS med', '#reduced');
    for kq = 1:K
        fprintf('%-10s %8.2f %8.4f %10.0f %9d%s\n', names{kq}, X2(kq), p(kq), median(essQ(:,kq)), nnz(reduced(:,kq)), flag(p(kq) < pMin));
    end
    fprintf('-> mode %s: %s (min p %.4f)\n\n', mode, pf(isPass(km)), min(p));

    fig = figure('Visible','off','Position',[0 0 1400 600]);
    [lo, hi] = binom_bounds(R, 1/nBins, 0.01);
    for kq = 1:K
        subplot(2, 5, kq); hold on;
        patch([0.5 nBins+0.5 nBins+0.5 0.5], [lo lo hi hi], [0.85 0.85 0.85], 'EdgeColor','none');
        bar(1:nBins, counts(:,kq), 1, 'FaceColor', [0.3 0.45 0.7]); yline(R/nBins, 'k--');
        title(sprintf('%s p=%.3f', names{kq}, p(kq)), 'Interpreter','none'); xlim([0.5 nBins+0.5]);
    end
    sgtitle(sprintf('T4.4 SBC Gaussian MRF, %s r=%d tau=%g (R=%d): grey = 99%% binomial band', mode, r, tau, R));
    exportgraphics(fig, fullfile(outDir, sprintf('T4_4_sbc_%s.png', mode))); close(fig);
    save(fullfile(outDir, sprintf('T4_4_sbc_%s.mat', mode)), 'ranks', 'essQ', 'reduced', 'names', 'p', 'X2', 'counts', 'R', 'Nv', 'tau', 'L');
end
fprintf('T4.4 overall: %s   (min p %.4f, threshold %.5f; total time %.1f min)\n', pf(all(isPass)), min(allP), pMin, toc(tStart)/60);

%% local functions
function [rk, e, isReduced] = sbc_rank(x, truth, L)
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
