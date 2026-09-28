%% test_T7_1_mixture_exact.m
%
% T7.1 (Phase 7): Gaussian-mixture hierarchical prior (K = 2) with FIXED mu_k, Sigma_k, pi on a
% linear Gaussian toy, where the per-voxel posterior is exactly a mixture of two Gaussians.
%
% Model (lingauss_fwd.m): y_i = A u_i + e_i, e_i ~ N(0, s^2 I), known s = 1, d = 2 ('linear',
%   lb = -Inf, ub = Inf, both under the hierarchy), m = 4 measurements, A = randn (seeded).
%   Prior: z_i ~ Categorical(pi), u_i | z_i = k ~ N(mu_k, Sigma_k), fixed
%       mu_1 = [-1.5; 0], mu_2 = [1.5; 1], Sigma_1 = [0.3 0.1; 0.1 0.2], Sigma_2 = [0.25 -0.05; -0.05 0.4],
%       pi = [0.6; 0.4].
%   Voxels: 150 drawn from the prior, plus 50 placed on the segment between mu_1 and mu_2
%   (t = 0.2..0.8), where the group membership is uncertain.
%   Known noise through the test-only fitting.fixedParams = struct('noise', s).
% Exact posterior per voxel (u and z):
%   component k: Q_k = A'A/s^2 + Sigma_k^-1, C_k = Q_k^-1, m_ik = C_k (A'y_i/s^2 + Sigma_k^-1 mu_k);
%   weights P(z_i = k | y_i) ∝ pi_k N(y_i | A mu_k, A Sigma_k A' + s^2 I);
%   E[u_i|y_i] = sum_k P_k m_ik, Cov[u_i|y_i] = sum_k P_k (C_k + m_ik m_ik') - E E'.
%
% Sampler: prior.hierarchical K = 2, fixed = true (run 4: collapsed mixture log-prior in the MH block, no z
%   draws; runs 1-3: z sampled every sweep, exact Gibbs), both update
%   schemes, adaptStepSize (burn-in only). 128 independent chains per voxel (run_chains.m, voxel copies;
%   valid because voxels are independent in fixed mode), starts u ~ N(0, 2^2) per copy.
%   20000 iterations, burn-in 4000, thinning 5 (3200 draws per chain).
%
% Test statistics per voxel (6 per voxel, 1200 per scheme), each estimated per chain c:
%   means E[u_p] (2), second moments E[(u_p - m_p)(u_q - m_q)] with the exact means (3), and the
%   sampler's output membership P(z = 1 | y) = out.hyper.membership (run 4: Rao-Blackwellised mean of
%   P(z = 1 | u, theta) over the kept samples; runs 1-3: frequency of z = 1) (1). MC error from the spread of the 128 independent chain estimates:
%       z = (mean_c est_c - exact) / (sd_c est_c / sqrt(128))
%   (if sd_c = 0, i.e. every chain gives the same value, |mean - exact| <= 1e-3 is required instead).
%   The ESS-based z of T3.1 (pooled draws, sd/sqrt(ESS)) is reported for information only.
% Strata (from the EXACT posterior only, reporting only in run 4), p_min = min_k P(z = k | y):
%   U (uncertain) p_min >= 0.02; N (negligible) p_min < 1e-6; R (rare switch) otherwise.
% Criteria (run 4 = the strict run-2 design), per scheme:
%   ALL 200 voxels x 6 statistics pooled (1200 z): frac(|z| > 1.96) <= 0.10 AND max|z| <= 4.5 (thresholds
%   as T3.1); if sd_c = 0, |mean - exact| <= 1e-3;
%   discriminative: median |z| of the sampler means against the exact means under a SINGLE Normal prior
%   with the mixture's mean and marginal covariance is >= 3 (a sampler that ignored the mixture fails).
%   The cache check (checkCache) is reported for the joint run.
%
% Revision 1 (27 Sep 2026, after the first run FAILED; thresholds unchanged, MC-error estimate changed).
%   First design: 16 chains, ESS-based z for the means/second moments and a Rao-Blackwell membership
%   statistic E[r_1(u)|y] (criterion frac <= 0.10, max <= 4.5), plus a between-chain t for the output
%   membership of uncertain voxels. Result: joint means max|z| 2.71, second moments max 5.09; componentwise
%   means max 13.1, second moments max 55.3; Rao-Blackwell max 2209 (voxels with P(z=1|y) = 1 to ~1e-6,
%   tiny sample SD); output membership passed (uncertain max|t| 2.17 / 2.72, confident max|f-P| 0.009).
%   Diagnosis: the worst voxels have a minor-group mass of 0.1-2% (e.g. componentwise voxel 149,
%   P(z=2|y) = 0.0087, 2 of 16 chains visited group 2); group switches are rare there, so the within-chain
%   ESS of 16 chains overstates the information. A targeted rerun of the 5 worst voxels with 400 chains x
%   40000 iterations matched the exact P(z=1|y) and E[u1] for both schemes (largest deviation 2.9 SE of
%   the 400-chain spread). The Rao-Blackwell statistic was dropped (its tail is never sampled when
%   P(z=1|y) ~ 1) in favour of testing the sampler's own membership output on every voxel.
% Revision 2 (27 Sep 2026, after revision 1 also FAILED; thresholds unchanged, voxels stratified).
%   Revision 1 (128 chains, between-chain SE, all 200 voxels pooled): joint frac 0.064, max|z| 40.9
%   (P(z=1)), C11 max 4.75; componentwise frac 0.058, max|z| 12.1 (P(z=1)), C11 max 4.86. All 69 uncertain
%   voxels passed every statistic (max|z| 2.65 joint, 2.75 componentwise); the failures were near-certain
%   voxels whose minor-group mass (~1e-6..1e-2) is sampled in O(1) excursions across the ensemble, where
%   the chain spread underestimates the MC error. The strata above are defined from the exact posterior
%   before this run.
% Run 3 (revision 2, stratified criteria; z-sampling sampler) PASSED: joint U+N (432 z) frac 0.042, max|z|
%   2.81, R membership max|f-P| 0.0020; componentwise U+N frac 0.021, max|z| 2.75, R max|f-P| 0.0025. In the
%   R stratum (128 voxels) the moments were only reported: max|z| C11 4.75 / 4.86, P(z=1) 40.9 / 12.1.
% Run 4 (27 Sep 2026): the sampler now uses the collapsed mixture log-prior (z summed out of the MH
%   target, no z draws in fixed mode) and the Rao-Blackwellised membership. The strict run-2 design is
%   reinstated unchanged (all 200 voxels, all 6 statistics, between-chain SE of 128 chains, same
%   thresholds); the strata are reported, not used in the criteria.
%   Result: FAIL (both schemes), only on the membership statistic. Moments pass on all 200 voxels incl. the
%   128 rare-switch voxels (joint max|z| 3.37, componentwise 3.91; before, R-stratum C11 max 4.75 / 4.86).
%   Membership: joint frac 0.105, max|z| 234; componentwise frac 0.095, max|z| 49.6 (all voxels pooled:
%   joint frac 0.067, componentwise 0.056). Uncertain voxels pass (max|z| 2.66 / 2.98). Failures are
%   near-certain voxels, estimate biased toward certainty by <= 1.4e-5 absolute (v29: 1.1e-3). Diagnosis
%   (scratch analysis): not precision (exact P with single-rounded y gives identical z); the sampler's
%   value equals E[P(z=1|u)] under the MAJOR posterior component alone, i.e. the minor u-mode (mass 1e-7 to
%   1e-2) is rarely or never visited by the random-walk MH chains, so the smooth RB estimate misses it and
%   the between-chain SD cannot see it. A mode-hopping limitation of u updates, not a target error.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T7_1_mixture_exact
%
% Kwok-Shing Chan @ MGH
% Date created: 27 September 2026
%

clearvars; tStart = tic;

% settings
d = 2; m = 4; s = 1.0; K = 2; Nchain = 128;
mu      = [-1.5 1.5; 0 1];
Sigma   = cat(3, [0.3 0.1; 0.1 0.2], [0.25 -0.05; -0.05 0.4]);
piW     = [0.6; 0.4];
Nprior  = 150; Nbetw = 50;
schemes = {'joint','componentwise'};
zFracMax = 0.10; zMaxMax = 4.5; zAltMin = 3; sd0Max = 1e-3;
pLo = 0.02; pHi = 0.98;     % reporting only (uncertain vs confident voxels)

seedA = 711; seedData = 712; seedStart = 713; seedRun = [714 715];
fprintf('T7.1 linear Gaussian, fixed K = 2 mixture prior: d=%d, m=%d, Nv=%d+%d, s=%.2f, %d chains\n', d, m, Nprior, Nbetw, s, Nchain);
fprintf('Seeds: A %d, data %d, start %d, run %s\n', seedA, seedData, seedStart, mat2str(seedRun));
fprintf(['Criteria per scheme (between-chain SE, %d chains): ALL voxels pooled frac(|z|>1.96) <= %.2f, max|z| <= %.1f ' ...
         '(sd = 0: |est-exact| <= %g); median|z_single| >= %.0f\n\n'], Nchain, zFracMax, zMaxMax, sd0Max, zAltMin);

rng(seedA);    A = randn(m, d);
rng(seedData);
z = 1 + (rand(1, Nprior) > piW(1)); u = zeros(d, Nprior);
for k = 1:K; u(:, z==k) = mu(:,k) + chol(Sigma(:,:,k),'lower')*randn(d, nnz(z==k)); end
t = linspace(0.2, 0.8, Nbetw);
u = [u, mu(:,1) + (mu(:,2) - mu(:,1)).*t];
Nv = size(u, 2);
y = A*u + s*randn(m, Nv);                              % [m, Nv]

% exact posterior (mixture) and the single-Normal alternative with the mixture's moments
[mPost, CPost, pz] = exact_post(y, A, s, mu, Sigma, piW);
mbar = mu*piW; V = -mbar*mbar.';
for k = 1:K; V = V + piW(k)*(Sigma(:,:,k) + mu(:,k)*mu(:,k).'); end
mAlt = exact_post(y, A, s, mbar, V, 1);
isUnc = pz(1,:) >= pLo & pz(1,:) <= pHi;
pMin  = min(pz, [], 1);
isU   = pMin >= 0.02; isN = pMin < 1e-6; isR = ~isU & ~isN;
fprintf('strata: U (p_min >= 0.02) %d, N (p_min < 1e-6) %d, R (rare switch) %d voxels\n', nnz(isU), nnz(isN), nnz(isR));
fprintf('exact P(z=1|y): %d uncertain voxels in [%.2f, %.2f] (%d of the %d between voxels), %d confident\n', ...
    nnz(isUnc), pLo, pHi, nnz(isUnc(end-Nbetw+1:end)), Nbetw, nnz(~isUnc));

f.modelParams = {'u1';'u2';'noise'};
f.lb = [-Inf; -Inf; 0]; f.ub = [Inf; Inf; 10]; f.xStepSize = [0.3; 0.3; 0.01];
f.algorithm = 'MH'; f.iteration = 20000; f.burnin = 4000; f.thinning = 5; f.metric = {'mean'};
f.fixedParams = struct('noise', s);
f.adaptStepSize = true; f.adaptInterval = 50;
f.prior.hierarchical = struct('K', K, 'fixed', true, 'mu', mu, 'Sigma', Sigma, 'pi', piW);

rng(seedStart);
x0.u1 = 2*randn(Nv*Nchain, 1); x0.u2 = 2*randn(Nv*Nchain, 1); x0.noise = s*ones(Nv*Nchain, 1);

pairs = [1 1; 1 2; 2 2];
isPass = false(1, numel(schemes));
for ks = 1:numel(schemes)
    f.updateScheme = schemes{ks};
    f.checkCache   = strcmp(schemes{ks}, 'joint');
    [post, out, tRun] = run_chains(y.', f, @(p) lingauss_fwd(p, A), x0, Nchain, seedRun(ks));
    U = cat(4, double(post.u1), double(post.u2));       % [Nv, Ns, Nchain, d]
    % per-chain estimates [Nv, Nchain] and exact values [Nv, 1]
    est = {}; ex = {}; nm = {};
    for p = 1:d
        est{end+1} = squeeze(mean(U(:,:,:,p), 2)); ex{end+1} = mPost(p,:).'; nm{end+1} = sprintf('E[u%d]', p); %#ok<SAGROW>
    end
    for kp = 1:size(pairs,1)
        p = pairs(kp,1); q = pairs(kp,2);
        tt = (U(:,:,:,p) - mPost(p,:).') .* (U(:,:,:,q) - mPost(q,:).');
        est{end+1} = squeeze(mean(tt, 2)); ex{end+1} = squeeze(CPost(p,q,:)); nm{end+1} = sprintf('C%d%d', p, q); %#ok<SAGROW>
    end
    memb = reshape(out.hyper.membership, Nv*Nchain, K);
    est{end+1} = reshape(memb(:,1), Nv, Nchain); ex{end+1} = pz(1,:).'; nm{end+1} = 'P(z=1)';
    Z = zeros(Nv, numel(est)); sd0Bad = 0; nSd0 = 0;
    for j = 1:numel(est)
        mj  = mean(est{j}, 2); sj = std(est{j}, 0, 2);
        Z(:,j) = (mj - ex{j}) ./ (sj ./ sqrt(Nchain));
        is0 = sj == 0;
        nSd0 = nSd0 + nnz(is0);
        sd0Bad = max([sd0Bad; abs(mj(is0) - ex{j}(is0))]);
        Z(is0, j) = 0;
    end
    zAlt = zeros(Nv, d);
    for p = 1:d
        zAlt(:,p) = (mean(est{p}, 2) - mAlt(p,:).') ./ (std(est{p}, 0, 2) ./ sqrt(Nchain));
    end
    % ESS-based z of the means (information only)
    zEss = zeros(Nv, d);
    for p = 1:d
        x = U(:,:,:,p); e = mcmc_bayes.ess(x);
        zEss(:,p) = (mean(reshape(x, Nv, []), 2) - mPost(p,:).') ./ (std(reshape(x, Nv, []), 0, 2) ./ sqrt(e));
    end
    passZ = mean(abs(Z(:)) > 1.96) <= zFracMax && max(abs(Z(:))) <= zMaxMax && sd0Bad <= sd0Max;
    passD = median(abs(zAlt(:))) >= zAltMin;
    isPass(ks) = passZ && passD;
    fprintf('%-13s: run %6.1f s | ALL pooled n=%d: frac|z|>1.96 %.3f, max|z| %.2f | sd=0 cases %d (max|est-exact| %.2g) -> %s\n', ...
        schemes{ks}, tRun, numel(Z), mean(abs(Z(:))>1.96), max(abs(Z(:))), nSd0, sd0Bad, pf(passZ));
    ZR = Z(isR,:);
    fprintf('               rare-switch stratum R (%d voxels, %d z): frac|z|>1.96 %.3f, max|z| %.2f\n', nnz(isR), numel(ZR), ...
        mean(abs(ZR(:))>1.96), max([abs(ZR(:)); 0]));
    for j = 1:numel(est)
        fprintf('               %-7s all: frac|z|>1.96 %.3f, max|z| %.2f | U: max|z| %.2f | N: max|z| %.2f | R: frac %.3f, max|z| %.2f\n', nm{j}, ...
            mean(abs(Z(:,j))>1.96), max(abs(Z(:,j))), max(abs(Z(isU,j))), max([abs(Z(isN,j)); 0]), ...
            mean(abs(Z(isR,j))>1.96), max([abs(Z(isR,j)); 0]));
    end
    fprintf('               sensitivity: median |z| vs single-Normal exact means %.1f -> %s\n', median(abs(zAlt(:))), pf(passD));
    fprintf('               info: ESS-based z of the means frac|z|>1.96 %.3f, max|z| %.2f | median ESS u1 %.0f | acceptance median %s\n', ...
        mean(abs(zEss(:))>1.96), max(abs(zEss(:))), median(mcmc_bayes.ess(U(:,:,:,1))), ...
        mat2str(median(reshape(out.diagnostics.acceptance, Nv*Nchain, []), 1), 3));
    if f.checkCache
        cc = out.diagnostics.cacheCheck;
        fprintf('               cache check (%d sweeps): max|diff| loglik %g, logprior %g, logjac %g\n', cc.Ncheck, cc.loglik, cc.logprior, cc.logjac);
    end
    fprintf('               -> %s\n', pf(isPass(ks)));
end
fprintf('\nT7.1 overall: %s   (total time %.1f s)\n', pf(all(isPass)), toc(tStart));

function s = pf(c)
if c; s = 'PASS'; else; s = 'FAIL'; end
end

% exact posterior of the linear Gaussian model under a Gaussian-mixture prior (K = 1: single Normal)
function [mP, CP, pz] = exact_post(y, A, s, mu, Sig, piW)
[m, Nv] = size(y); d = size(A,2); K = numel(piW);
mk = zeros(d, Nv, K); Ck = zeros(d, d, K); lw = zeros(K, Nv);
for k = 1:K
    P = inv(Sig(:,:,k)); Q = A.'*A/s^2 + P; C = inv(Q); Ck(:,:,k) = (C + C.')/2;
    mk(:,:,k) = C*(A.'*y/s^2 + P*mu(:,k));
    M = A*Sig(:,:,k)*A.' + s^2*eye(m); r = y - A*mu(:,k); L = chol(M, 'lower'); a = L\r;
    lw(k,:) = log(piW(k)) - sum(log(diag(L))) - 0.5*sum(a.^2, 1);
end
pz = exp(lw - max(lw, [], 1)); pz = pz ./ sum(pz, 1);
mP = zeros(d, Nv); CP = zeros(d, d, Nv);
for k = 1:K; mP = mP + pz(k,:).*mk(:,:,k); end
for i = 1:Nv
    S2 = zeros(d);
    for k = 1:K; S2 = S2 + pz(k,i)*(Ck(:,:,k) + mk(:,i,k)*mk(:,i,k).'); end
    CP(:,:,i) = S2 - mP(:,i)*mP(:,i).';
end
end
