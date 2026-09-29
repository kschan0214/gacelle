%% test_T9b_0_studentt_exact.m
%
% T9b.0 (Phase 9b): Student-t population prior with FIXED mu, Sigma, nu on a 1D linear Gaussian toy,
% where the per-voxel posterior is known up to a 1D numerical integration.
%
% Model (lingauss_fwd.m): y_i = a u_i + e_i, e_i ~ N(0, s^2 I), known s = 1 (test-only
%   fitting.fixedParams), d = 1 ('linear', lb = -Inf, ub = Inf, under the hierarchy), m = 4 measurements,
%   a = randn (seeded) scaled to |a| = 2, so the likelihood SD of u is s/|a| = 0.5.
%   Prior u_i ~ t_nu(mu, sg^2), fixed nu = 4, mu = 0.5, sg = 1 (prior.hierarchical.distribution = 't',
%   fixed = true): the sampler uses the collapsed t log-prior (no lambda).
%   Voxels: 100 drawn from the prior, plus 50 "tail" voxels with true u = mu + 2.5..9 (the t prior pulls
%   them much less than a Normal prior of the same scale would; their posterior is skewed).
% Exact posterior per voxel: p(u | y) ∝ exp(-(u - uhat)^2/(2 sl^2)) (1 + (u-mu)^2/(nu sg^2))^(-(nu+1)/2),
%   uhat = a'y/|a|^2, sl = s/|a|; CDF by the trapezoid rule on 2e5 nodes over uhat/mu +- 12 scales
%   (grid_quantiles.m).
%
% Sampler: both update schemes, adaptStepSize (burn-in only), 64 independent chains per voxel
%   (run_chains.m, voxel copies; valid because voxels are independent in fixed mode), starts
%   u ~ mu + 3 N(0,1) per copy. 20000 iterations, burn-in 4000, thinning 5 (3200 draws per chain).
%
% Test statistics per voxel: the 5%, 25%, 50%, 75%, 95% posterior quantiles and the posterior mean
%   (6 per voxel, 900 per scheme). Estimate: pooled over all chains; MC error from the spread of the
%   64 per-chain estimates:  z = (pooled - exact) / (sd_c(per-chain) / sqrt(64)).
% Criteria (stated before running, thresholds as T3.1/T7.1), per scheme:
%   PASS if frac(|z| > 1.96) <= 0.10 AND max|z| <= 4.5, AND the test is discriminative: the median |z|
%   of the tail voxels' sampler medians against the exact medians under a NORMAL prior N(mu, sg^2) is
%   >= 3 (a sampler that used the Normal instead of the t fails).
%   The cache check (checkCache) is reported for the joint run.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T9b_0_studentt_exact
%
% Kwok-Shing Chan @ MGH
% Date created: 29 September 2026
%

clearvars; tStart = tic;

% settings
nu = 4; mu = 0.5; sg = 1; m = 4; s = 1; Nchain = 64;
Ntyp = 100; uTail = linspace(2.5, 9, 50) + mu;
schemes = {'joint','componentwise'};
qs = [0.05 0.25 0.5 0.75 0.95];
zFracMax = 0.10; zMaxMax = 4.5; zNormalMin = 3;
seedA = 941; seedData = 942; seedStart = 943; seedRun = [944 945];

fprintf('T9b.0 1D linear Gaussian, fixed Student-t prior: nu=%g, mu=%g, sg=%g, m=%d, s=%g, %d chains\n', nu, mu, sg, m, s, Nchain);
fprintf('Seeds: a %d, data %d, start %d, run %s\n', seedA, seedData, seedStart, mat2str(seedRun));
fprintf('Criterion per scheme: frac(|z|>1.96) <= %.2f, max|z| <= %.1f, median|z_normal| (tail voxels) >= %.0f\n\n', zFracMax, zMaxMax, zNormalMin);

rng(seedA);    a = randn(m, 1); a = 2*a/norm(a);
rng(seedData);
uT = [mu + sg*randn(1, Ntyp)./sqrt(2*randg(nu/2, 1, Ntyp)/nu), uTail];     % t_nu draws (Gamma scale mixture), then the tail voxels
Nv = numel(uT); isTail = (1:Nv) > Ntyp;
y  = (a*uT + s*randn(m, Nv)).';                                           % [Nv, m]

% exact posterior quantiles and mean (t prior), and the Normal-prior median (discrimination)
uh = (y*a).'/(a.'*a); sl = s/norm(a);
qx = zeros(Nv, numel(qs)); mx = zeros(Nv, 1);
for v = 1:Nv
    x  = linspace(min(uh(v), mu) - 12*max(sl, sg), max(uh(v), mu) + 12*max(sl, sg), 2e5);
    lp = -(uh(v) - x).^2/(2*sl^2) - (nu+1)/2*log1p((x - mu).^2/(nu*sg^2));
    p  = exp(lp - max(lp));
    qx(v,:) = grid_quantiles(x, p, qs);
    mx(v)   = trapz(x, x.*p)/trapz(x, p);
end
cN = 1/(1/sl^2 + 1/sg^2); medNormal = cN*(uh/sl^2 + mu/sg^2);            % Normal prior: Gaussian posterior
fprintf('tail voxels: exact median minus Normal-prior median, range [%.2f, %.2f]\n', min(qx(isTail,3) - medNormal(isTail).'), max(qx(isTail,3) - medNormal(isTail).'));

f.modelParams = {'u1';'noise'};
f.lb = [-Inf; 0]; f.ub = [Inf; 10]; f.xStepSize = [0.3; 0.01];
f.algorithm = 'MH'; f.iteration = 20000; f.burnin = 4000; f.thinning = 5; f.metric = {'mean'};
f.fixedParams = struct('noise', s);
f.adaptStepSize = true; f.adaptInterval = 50;
f.prior.hierarchical = struct('distribution', 't', 'nu', nu, 'fixed', true, 'mu', mu, 'Sigma', sg^2);

rng(seedStart);
x0.u1 = mu + 3*randn(Nv*Nchain, 1);
x0.noise = s*ones(Nv*Nchain, 1);

isPass = false(1, numel(schemes));
for ks = 1:numel(schemes)
    f.updateScheme = schemes{ks};
    f.checkCache   = strcmp(schemes{ks}, 'joint');
    [post, out, tRun] = run_chains(y, f, @(p) lingauss_fwd(p, a), x0, Nchain, seedRun(ks));
    U  = double(post.u1);                                                % [Nv, Ns, Nchain]
    Up = reshape(U, Nv, []);
    z  = zeros(Nv, numel(qs) + 1);
    for k = 1:numel(qs)
        qc = squeeze(sample_quantile(U, qs(k), 2));                      % [Nv, Nchain]
        z(:,k) = (sample_quantile(Up, qs(k), 2) - qx(:,k)) ./ (std(qc, 0, 2)/sqrt(Nchain));
        if k == 3
            zN = (sample_quantile(Up, qs(k), 2) - medNormal(:)) ./ (std(qc, 0, 2)/sqrt(Nchain));
        end
    end
    mc = squeeze(mean(U, 2));
    z(:,end) = (mean(Up, 2) - mx) ./ (std(mc, 0, 2)/sqrt(Nchain));
    isPass(ks) = mean(abs(z(:)) > 1.96) <= zFracMax && max(abs(z(:))) <= zMaxMax && median(abs(zN(isTail))) >= zNormalMin;
    fprintf('%-13s: run %6.1f s | all %d z: frac|z|>1.96 %.3f, max|z| %.2f | typical: frac %.3f max %.2f | tail: frac %.3f max %.2f\n', ...
        schemes{ks}, tRun, numel(z), mean(abs(z(:)) > 1.96), max(abs(z(:))), mean(abs(z(~isTail,:)) > 1.96, 'all'), max(abs(z(~isTail,:)), [], 'all'), ...
        mean(abs(z(isTail,:)) > 1.96, 'all'), max(abs(z(isTail,:)), [], 'all'));
    fprintf('               per statistic max|z| (q05 q25 q50 q75 q95 mean): %s\n', mat2str(max(abs(z), [], 1), 3));
    fprintf('               sensitivity: median |z| of the tail medians vs the Normal-prior medians %.1f | acceptance median %.3f\n', ...
        median(abs(zN(isTail))), median(out.diagnostics.acceptance(:)));
    if f.checkCache
        cc = out.diagnostics.cacheCheck;
        fprintf('               cache check (%d sweeps): max|diff| loglik %g, logprior %g, logjac %g\n', cc.Ncheck, cc.loglik, cc.logprior, cc.logjac);
    end
    fprintf('               -> %s\n', pf(isPass(ks)));
end
fprintf('\nT9b.0 overall: %s   (total time %.1f s)\n', pf(all(isPass)), toc(tStart));

function s = pf(c)
if c; s = 'PASS'; else; s = 'FAIL'; end
end

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
