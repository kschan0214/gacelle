function R = gmrf_exact_run(cfg)
% GMRF_EXACT_RUN One configuration of the exact Gaussian-MRF test (T4.2 / T4.3).
%
%   R = gmrf_exact_run(cfg)
%
% Model: linear Gaussian toy y_i = A u_i + e_i, e_i ~ N(0, s^2 I), known s (test-only
%   fitting.fixedParams), d = 2, m = 4, on an 8 x 8 x 4 grid with a mask (~15% holes, seeded).
%   Prior: u_i ~ N(mu, Sigma) (fixed) times exp(-Phi_MRF), quadratic potential rho = x^2/2,
%   W = 1./sqrt(diag(Sigma)) (the default), fixed symmetric edge weights w_ij in [0.5, 1.5]
%   (a function of the unordered voxel pair only).
% Exact joint posterior (voxel-major ordering, index (i-1)*d + p), double, host:
%   Q = I (x) (A'A/s^2 + Sigma^-1) + (1/tau) L_w (x) diag(W),  b = I (x) A'/s^2 y + 1 (x) Sigma^-1 mu,
%   mean = Q \ b, C = inv(Q) (dense, n*d ~ 460). L_w = D - A_w is built here by brute-force
%   enumeration of the masked voxel pairs (independent of mcmc_bayes.build_neighbours).
%   The scaling (1/tau) L_w (x) diag(W) follows from sum_{(i,j) in E} w_ij (u_i - u_j)^2 = u' L_w u
%   with each edge counted once (see the mcmc_bayes header); it is verified numerically below
%   (Phi from the sampler's local function summed over a colour sweep vs u'Qmrf u/2).
% Sampler: Nchain chains in one call, as Nchain copies of the volume stacked along dim 3 and
%   separated by an empty slice (no edge crosses between copies; in '2d' mode slices are
%   independent anyway). Starts u ~ mu + 2 sd .* randn per chain (over-dispersed).
%
% cfg fields: mode, radius, connectivity, tau, updateScheme, mrfUpdate ('chromatic' |
%   'simultaneous' TEST ONLY), iteration, burnin, thinning, Nchain, seedRun
%
% R: structure with the z-scores per category, their summary, the exact and sampler moments,
%   ratios, ESS, R-hat, acceptance, runtime.
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%

d = 2; m = 4; s = 1; dims = [8 8 4];
mu      = [0.5; -1];
sds     = [0.8; 0.5];
Sigma   = diag(sds) * [1 0.3; 0.3 1] * diag(sds);
W       = 1 ./ sqrt(diag(Sigma));

% fixed geometry, design and data (independent of the configuration)
rng(91); mask = rand(dims) > 0.15;
rng(92); A = randn(m, d);
mask_idx = find(mask); Nv = numel(mask_idx);
rng(93); u0 = mu + chol(Sigma,'lower')*randn(d, Nv);
y  = A*u0 + s*randn(m, Nv);                             % [m, Nv]

% brute-force adjacency and weighted Laplacian
[i, j, k] = ind2sub(dims, mask_idx);
P   = [i j k];
D   = abs(permute(P, [1 3 2]) - permute(P, [3 1 2]));
if strcmp(cfg.connectivity, 'face'); Adj = sum(D, 3) == 1; else; Adj = max(D, [], 3) <= cfg.radius & max(D, [], 3) > 0; end
if strcmp(cfg.mode, '2d'); Adj = Adj & D(:,:,3) == 0; end
[a, b] = ndgrid(1:Nv, 1:Nv);
Wadj = Adj .* pair_weight(min(a, b), max(a, b));
L    = diag(sum(Wadj, 2)) - Wadj;
[ea, eb] = find(triu(Adj));                             % edges, each once

% exact posterior
Pr   = inv(Sigma);
Qmrf = kron(sparse(L), sparse(diag(W))) ./ cfg.tau;
Q    = kron(speye(Nv), sparse(A.'*A/s^2 + Pr)) + Qmrf;
bvec = reshape(A.'*y/s^2 + Pr*mu, [], 1);
mPost = reshape(Q \ bvec, d, Nv);
C    = inv(full(Q)); C = (C + C.')/2;
% no-MRF exact means (discrimination check)
Q0   = kron(speye(Nv), sparse(A.'*A/s^2 + Pr));
m0   = reshape(Q0 \ bvec, d, Nv);

% stacked volume with Nchain copies separated by an empty slice
Nc   = cfg.Nchain;
Z    = Nc*dims(3) + (Nc-1);
mask4 = false([dims(1:2) Z]);
yy   = zeros([dims(1:2) Z m]);
yimg = zeros(numel(mask), m); yimg(mask_idx, :) = y.';
yimg = reshape(yimg, [dims m]);
rng(cfg.seedRun);
for c = 1:Nc
    zs = (c-1)*(dims(3)+1) + (1:dims(3));
    mask4(:,:,zs) = mask;
    yy(:,:,zs,:)  = yimg;
end
mask4_idx = find(mask4);
for p = 1:d
    x0.(sprintf('u%d',p)) = zeros(size(mask4));
    x0.(sprintf('u%d',p))(mask4_idx) = mu(p) + 2*sds(p)*randn(Nv*Nc, 1);
end

% edge weights in the sampler layout: same function of the within-copy voxel pair
nbr4 = mcmc_bayes.build_neighbours(mask4_idx, size(mask4), cfg.mode, cfg.radius, cfg.connectivity);
ew   = zeros(size(nbr4));
[kk, vv] = find(nbr4 > 0);
nn   = double(nbr4(sub2ind(size(nbr4), kk, vv)));
va   = mod(vv-1, Nv) + 1; vb = mod(nn-1, Nv) + 1;
ew(sub2ind(size(nbr4), kk, vv)) = pair_weight(min(va, vb), max(va, vb));

% check of the Laplacian scaling with the sampler's own local function (double, host):
% moving all voxels of one colour, sum_i dPhi_i must equal Phi(u') - Phi(u) = (u'Qmrf u' - u Qmrf u)/2
nbr1 = mcmc_bayes.build_neighbours(mask_idx, dims, cfg.mode, cfg.radius, cfg.connectivity);
col1 = mcmc_bayes.build_colours(mask_idx, dims, cfg.mode, cfg.radius, cfg.connectivity, nbr1);
ew1  = zeros(size(nbr1)); [k1, v1] = find(nbr1 > 0); n1 = double(nbr1(sub2ind(size(nbr1), k1, v1)));
ew1(sub2ind(size(nbr1), k1, v1)) = pair_weight(min(v1, n1), max(v1, n1));
self = repmat(int32(1:Nv), size(nbr1,1), 1); nS = nbr1; nS(nbr1 == 0) = self(nbr1 == 0);
rng(94); uT = randn(d, Nv);
R.scalingCheck = 0;
for c = unique(col1)
    act = find(col1 == c);
    uN  = uT; uN(:,act) = uN(:,act) + randn(d, numel(act));
    loc = sum(mcmc_bayes.mrf_local_delta(uN(:,act), uT(:,act), uT, nS(:,act), ew1(:,act), W/cfg.tau, [1;1], 'quadratic'));
    ref = 0.5*(uN(:).'*Qmrf*uN(:) - uT(:).'*Qmrf*uT(:));
    R.scalingCheck = max(R.scalingCheck, abs(loc - ref)/abs(ref));
end

% sampler
f.modelParams = {'u1';'u2';'noise'};
f.lb = [-Inf; -Inf; 0]; f.ub = [Inf; Inf; 10]; f.xStepSize = [0.3; 0.3; 0.01];
f.algorithm = 'MH'; f.iteration = cfg.iteration; f.burnin = cfg.burnin; f.thinning = cfg.thinning;
f.metric = {'mean'}; f.repetition = 1;
f.fixedParams = struct('noise', s);
f.adaptStepSize = true; f.adaptInterval = 50;
f.updateScheme = cfg.updateScheme;
f.mrfUpdate = cfg.mrfUpdate;
f.prior.hierarchical = struct('fixed', true, 'mu', mu, 'Sigma', Sigma);
f.prior.mrf = struct('potential', 'quadratic', 'tau', cfg.tau, 'mode', cfg.mode, 'radius', cfg.radius, ...
                     'connectivity', cfg.connectivity, 'edgeWeights', ew);
rng(cfg.seedRun); parallel.gpu.rng(cfg.seedRun);
t0 = tic;
evalc('out = mcmc_bayes().optimisation(yy, mask4, [], x0, f, @(pp) lingauss_fwd(pp, A));');
R.tRun = toc(t0);

Ns = size(out.posterior.u1, 2);
U  = zeros(Nv, Ns, Nc, d);
for p = 1:d
    U(:,:,:,p) = permute(reshape(double(out.posterior.(sprintf('u%d',p))), Nv, Nc, Ns), [1 3 2]);
end

% z-scores: means, marginal variances, within-voxel covariance, neighbour covariances (p = q)
Cv  = @(v, p, w, q) C(sub2ind(size(C), (v-1)*d + p, (w-1)*d + q));
zM  = zeros(Nv, d); zF = zeros(Nv, d); zV = zeros(Nv, d); rV = zeros(Nv, d);
for p = 1:d
    x = U(:,:,:,p);
    [zM(:,p), ~] = zstat(x, mPost(p,:).');
    zF(:,p) = zstat(x, m0(p,:).');
    t = (x - mPost(p,:).').^2;
    cref = Cv((1:Nv).', p, (1:Nv).', p);
    [zV(:,p), mt] = zstat(t, cref);
    rV(:,p) = mt ./ cref;
end
t   = (U(:,:,:,1) - mPost(1,:).') .* (U(:,:,:,2) - mPost(2,:).');
zX  = zstat(t, Cv((1:Nv).', 1, (1:Nv).', 2));
Ne  = numel(ea);
zN  = zeros(Ne, d); rN = zeros(Ne, d); cN = zeros(Ne, d);
for p = 1:d
    t = (U(ea,:,:,p) - mPost(p,ea).') .* (U(eb,:,:,p) - mPost(p,eb).');
    cN(:,p) = Cv(ea, p, eb, p);
    [zN(:,p), mt] = zstat(t, cN(:,p));
    rN(:,p) = mt ./ cN(:,p);
end
% neighbour covariances vs 0 (the value without the MRF), information only
zN0 = zeros(Ne, d);
for p = 1:d
    t = (U(ea,:,:,p) - mPost(p,ea).') .* (U(eb,:,:,p) - mPost(p,eb).');
    zN0(:,p) = zstat(t, zeros(Ne,1));
end

z = [zM(:); zV(:); zX(:); zN(:)];
R.z = z; R.N = numel(z);
R.cat = struct('name', {'means','variances','within-voxel cov','neighbour cov'}, ...
               'z', {zM(:), zV(:), zX(:), zN(:)});
R.zFlatMedian   = median(abs(zF(:)));
R.zN0Median     = median(abs(zN0(:)));
R.varRatio      = [median(rV(:)), min(rV(:)), max(rV(:))];
R.nbrCovRatio   = [median(rN(:)), min(rN(:)), max(rN(:))];
R.nbrCorrExact  = median(cN(:,1) ./ sqrt(Cv(ea,1,ea,1).*Cv(eb,1,eb,1)));
Rh = zeros(Nv, d); Es = zeros(Nv, d);
for p = 1:d; Rh(:,p) = mcmc_bayes.rhat(U(:,:,:,p)); Es(:,p) = mcmc_bayes.ess(U(:,:,:,p)); end
R.rhatMax   = max(Rh(:));
R.essMedian = median(Es(:)); R.essMin = min(Es(:));
acc = reshape(out.diagnostics.acceptance, numel(mask4), []);
R.accMedian = median(acc(mask4(:), :), 1);
R.Nv = Nv; R.Nedges = Ne; R.settings = out.settings.mrf;
R.priorVsMrfDiag = [mean(diag(A.'*A/s^2 + Pr)), mean(full(diag(Qmrf)))];   % mean diagonal: local vs MRF precision
end

function w = pair_weight(a, b)
% symmetric edge weight in [0.5, 1.5) of the unordered pair (a <= b)
w = 0.5 + mod(a*7919 + b*104729, 101)/101;
end

function [z, mt] = zstat(t, target)
% z = (mean(t) - target) / (sd(t)/sqrt(ESS(t))), per row; t is [n, Ns, Nchain]; chunked for memory
n  = size(t, 1);
z  = zeros(n, 1); mt = zeros(n, 1);
for kc = 1:500:n
    idx = kc:min(kc+499, n);
    tt  = t(idx,:,:);
    e   = mcmc_bayes.ess(tt);
    tf  = reshape(tt, numel(idx), []);
    mt(idx) = mean(tf, 2);
    z(idx)  = (mt(idx) - target(idx)) ./ (std(tf, 0, 2) ./ sqrt(e));
end
end
