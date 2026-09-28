%% test_T3_1_linear_gaussian_exact.m
%
% T3.1 (Phase 3): hierarchical Normal prior with FIXED mu, Sigma on a linear
% Gaussian toy, where the per-voxel posterior is exactly Gaussian.
%
% Model (lingauss_fwd.m): y_i = A u_i + e_i, e_i ~ N(0, s^2 I), known s,
%   d = 3 parameters u1..u3 ('linear' transform, lb = -Inf, ub = Inf, all under
%   the hierarchy), m = 8 measurements, A fixed (randn, seeded), s = 1.5.
%   Prior u_i ~ N(mu, Sigma) with fixed mu = [0.5 -1 2], SDs [0.6 0.4 0.8] and
%   correlations r12 = 0.5, r13 = -0.3, r23 = 0.2. The likelihood precision
%   A'A/s^2 is of the same order as Sigma^-1, so the prior moves the posterior.
%   Truth: 200 voxels drawn from the prior.
%   Known noise: 'noise' is removed from the sampled set with the test-only
%   fitting.fixedParams = struct('noise', s) (value injected before every
%   likelihood call, like the S0Param injection). Likelihood 'gaussian'.
% Exact posterior per voxel: Q = A'A/s^2 + Sigma^-1, C = Q^-1,
%   m_i = C (A'y_i/s^2 + Sigma^-1 mu).
%
% Sampler: prior.hierarchical fixed = true, both update schemes ('joint' and
%   'componentwise'), adaptStepSize (burn-in only), 4 independent chains per voxel
%   (run_chains.m, voxel copies; valid because voxels are independent in fixed
%   mode), starts u ~ N(0, 2^2) per chain copy (over-dispersed).
%   30000 iterations, burn-in 6000, thinning 5 (4800 draws per chain).
%
% Test statistics per voxel (9 per voxel, 1800 per scheme):
%   means:        z = (mean(u_p) - m_ip) / (sd(u_p)/sqrt(ESS(u_p)))
%   covariances:  t = (u_p - m_ip)(u_q - m_iq) (exact means, so E[t] = C_pq),
%                 z = (mean(t) - C_pq) / (sd(t)/sqrt(ESS(t))), p <= q
%   ESS: mcmc_bayes.ess (multi-chain, split).
% Criterion (stated before running), per scheme:
%   PASS if frac(|z| > 1.96) <= 0.10 AND max|z| <= 4.5 (nominal 0.05; slack for
%   correlated statistics within a voxel and ESS-estimation noise; under
%   independence P(max of 1800 |N(0,1)| > 4.5) ~ 0.012), AND the test is
%   discriminative: the median |z| of the sampler means against the FLAT-prior
%   exact means (C_flat A'y/s^2) is >= 3 (i.e. a sampler that ignored the
%   prior term would fail).
%   Split-R-hat (4 chains) is reported; the cache check (checkCache) is reported
%   for the joint run.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T3_1_linear_gaussian_exact
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%

clearvars; tStart = tic;

% settings
d = 3; m = 8; Nv = 200; s = 1.5; Nchain = 4;
mu      = [0.5; -1; 2];
sds     = [0.6; 0.4; 0.8];
R       = [1 0.5 -0.3; 0.5 1 0.2; -0.3 0.2 1];
Sigma   = diag(sds)*R*diag(sds);
schemes = {'joint','componentwise'};
zFracMax = 0.10; zMaxMax = 4.5; zFlatMin = 3;

seedA = 71; seedData = 72; seedStart = 73; seedRun = [74 75];
fprintf('T3.1 linear Gaussian, fixed hierarchical prior: d=%d, m=%d, Nv=%d, s=%.2f, %d chains\n', d, m, Nv, s, Nchain);
fprintf('Seeds: A %d, data %d, start %d, run %s\n', seedA, seedData, seedStart, mat2str(seedRun));
fprintf('Criterion per scheme: frac(|z|>1.96) <= %.2f, max|z| <= %.1f, median|z_flat| >= %.0f\n\n', zFracMax, zMaxMax, zFlatMin);

rng(seedA);    A = randn(m, d);
rng(seedData); u = mu + chol(Sigma,'lower')*randn(d, Nv);
y = (A*u + s*randn(m, Nv)).';                           % [Nv, m]

% exact posterior
P     = inv(Sigma);
Q     = A.'*A/s^2 + P;   C = inv(Q);
mPost = C*(A.'*y.'/s^2 + P*mu);                         % [d, Nv]
mFlat = (A.'*A) \ (A.'*y.');                            % flat prior in u
fprintf('prior SD %s, posterior SD %s, flat-prior posterior SD %s\n', mat2str(sds.',3), mat2str(sqrt(diag(C)).',3), ...
    mat2str(sqrt(diag(inv(A.'*A/s^2))).',3));

f.modelParams = [arrayfun(@(k) sprintf('u%d',k), (1:d).', 'UniformOutput', false); {'noise'}];
f.lb = [-Inf(d,1); 0]; f.ub = [Inf(d,1); 10]; f.xStepSize = [0.3*ones(d,1); 0.01];
f.algorithm = 'MH'; f.iteration = 30000; f.burnin = 6000; f.thinning = 5; f.metric = {'mean'};
f.fixedParams = struct('noise', s);
f.adaptStepSize = true; f.adaptInterval = 50;
f.prior.hierarchical = struct('fixed', true, 'mu', mu, 'Sigma', Sigma);

rng(seedStart);
for k = 1:d; x0.(sprintf('u%d',k)) = 2*randn(Nv*Nchain, 1); end
x0.noise = s*ones(Nv*Nchain, 1);

pairs = [1 1; 1 2; 1 3; 2 2; 2 3; 3 3];
isPass = false(1, numel(schemes));
for ks = 1:numel(schemes)
    f.updateScheme = schemes{ks};
    f.checkCache   = strcmp(schemes{ks}, 'joint');
    [post, out, tRun] = run_chains(y, f, @(p) lingauss_fwd(p, A), x0, Nchain, seedRun(ks));
    U = cat(4, post.u1, post.u2, post.u3);             % [Nv, Ns, Nchain, d]
    zMean = zeros(Nv, d); zFlat = zeros(Nv, d); zCov = zeros(Nv, size(pairs,1));
    for p = 1:d
        x   = U(:,:,:,p);
        e   = mcmc_bayes.ess(x);
        sdx = std(reshape(x, Nv, []), 0, 2);
        mx  = mean(reshape(x, Nv, []), 2);
        zMean(:,p) = (mx - mPost(p,:).') ./ (sdx ./ sqrt(e));
        zFlat(:,p) = (mx - mFlat(p,:).') ./ (sdx ./ sqrt(e));
    end
    for kp = 1:size(pairs,1)
        p = pairs(kp,1); q = pairs(kp,2);
        t = (U(:,:,:,p) - mPost(p,:).') .* (U(:,:,:,q) - mPost(q,:).');
        e = mcmc_bayes.ess(t);
        zCov(:,kp) = (mean(reshape(t, Nv, []), 2) - C(p,q)) ./ (std(reshape(t, Nv, []), 0, 2) ./ sqrt(e));
    end
    z   = [zMean(:); zCov(:)];
    Rh  = mcmc_bayes.rhat(U(:,:,:,1)); for p = 2:d; Rh = max(Rh, mcmc_bayes.rhat(U(:,:,:,p))); end
    isPass(ks) = mean(abs(z) > 1.96) <= zFracMax && max(abs(z)) <= zMaxMax && median(abs(zFlat(:))) >= zFlatMin;
    fprintf('%-13s: run %6.1f s | means: frac|z|>1.96 %.3f, max|z| %.2f | cov: frac %.3f, max|z| %.2f | pooled n=%d frac %.3f max %.2f | median z mean %+.3f\n', ...
        schemes{ks}, tRun, mean(abs(zMean(:))>1.96), max(abs(zMean(:))), mean(abs(zCov(:))>1.96), max(abs(zCov(:))), ...
        numel(z), mean(abs(z)>1.96), max(abs(z)), median(zMean(:)));
    fprintf('               sensitivity: median |z| vs flat-prior means %.1f | R-hat max %.4f | acceptance median %s\n', ...
        median(abs(zFlat(:))), max(Rh), mat2str(median(reshape(out.diagnostics.acceptance, Nv*Nchain, []), 1), 3));
    if f.checkCache
        cc = out.diagnostics.cacheCheck;
        fprintf('               cache check (%d sweeps): max|diff| loglik %g, logprior %g, logjac %g\n', cc.Ncheck, cc.loglik, cc.logprior, cc.logjac);
    end
    fprintf('               -> %s\n', pf(isPass(ks)));
end
fprintf('\nT3.1 overall: %s   (total time %.1f s)\n', pf(all(isPass)), toc(tStart));

function s = pf(c)
if c; s = 'PASS'; else; s = 'FAIL'; end
end
