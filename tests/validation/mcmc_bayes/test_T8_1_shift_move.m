%% test_T8_1_shift_move.m
%
% T8.1: joint location-shift move of the free hierarchical prior (prior.hierarchical.shiftMove).
% Correctness (the move leaves the posterior unchanged) and mixing gain, on a weakly identified
% linear Gaussian hierarchical toy in the slow regime (per-voxel likelihood much wider than Sigma).
%
% Designs (both run by default; env MCMC_BAYES_T8_PARTS, e.g. 'A2,B2', selects a subset):
%   A1/B1 : A = randn(4,2) (seeded, ANISOTROPIC: likelihood SD [1.19 0.30]), ratio 4 (first design)
%   A2/B2 : A = [1 0; 0 1; 0.7 0.7; 0.7 -0.7] (isotropic, A'A = 1.98 I), ratio 3; B2 groups further apart
%   (A2/B2 were added after A1/B1 were inconclusive, see the notes below; criteria unchanged)
%
% Part A (K = 1, exact reference)
%   Model (lingauss_fwd.m): y_i = A u_i + e_i, e_i ~ N(0, s^2 I), known s (test-only fixedParams),
%   d = 2, m = 4, n = 2000 voxels. Truth u_i ~ N(mu, Sigma),
%   mu = [0.5 -0.3], Sigma = [0.04 0.012; 0.012 0.03]. s is set so that the mean per-voxel
%   likelihood SD sqrt(diag(s^2 (A'A)^-1)) is 'ratio' x the mean prior SD (slow regime).
%   Hyperprior (explicit, so the reference is fully specified): NIW, m0 = [0 0], kappa0 = 1e-3,
%   nu0 = 4, Psi0 = 0.05 I.
%   Reference: u integrates out analytically, y_i | mu, Sigma ~ N(A mu, M), M = A Sigma A' + s^2 I, so
%       log p(mu, Sigma | y) = -n/2 log|M| - 1/2 tr(M^-1 [S_yy + n (ybar - A mu)(ybar - A mu)'])
%                              + log N(mu | m0, Sigma/kappa0) + log IW(Sigma | Psi0, nu0) + const,
%   a 5-dim posterior, sampled on the host in double by random-walk MH on (mu, log L11, L21, log L22),
%   Sigma = L L' (log-Jacobian 3 log L11 + 2 log L22), proposal covariance from pilot runs,
%   4 chains x 250000 kept iterations (thinned by 10). Voxel reference (100-voxel subset, both u):
%   Rao-Blackwellised over the reference draws, u_i | y, mu, Sigma ~ N(C (A'y_i/s^2 + Sigma^-1 mu), C),
%   C = (A'A/s^2 + Sigma^-1)^-1: E[u_i] = E[m_i], Var[u_ip] = E[C_pp] + Var[m_ip].
%   Reference validity (else the script errors): split-R-hat <= 1.01 and ESS >= 20000 for every
%   mu/Sigma entry.
%   Sampler arms (identical except the move): mcmc_bayes, free NIW (above), joint update, adaptStepSize
%   (interval 50, burn-in only), 4 repetitions (overdisp 0.01, start u = 0), 40000 iterations, burn-in
%   10000, thinning 10 (3000 kept per chain); arm 'noshift' (shiftMove absent) and arm 'shift'
%   (shiftMove = true, shiftEvery = 1, default scale).
%
% Test statistics per arm (z against the reference; SE from the multi-chain ESS of the arm, reference
%   MC error added in quadrature, except for the voxel SDs, whose reference MC error is neglected):
%   means: z = (mean - ref) / sqrt(se^2 + seRef^2),  se = sd / sqrt(ESS)
%   SDs:   z = (sd - refSD) / sqrt(seS^2 + seSRef^2),  seS = sqrt(var((x-m)^2)/ESS((x-m)^2)) / (2 sd)
%   (H) hyper: mean and SD of mu1, mu2, Sigma11, Sigma21, Sigma22  -> 10 z
%   (V) voxels: mean and SD of u1, u2 of voxels 1..100            -> 400 z
% Criterion (stated before running), per arm:
%   conclusive only if min ESS over the 5 hyper entries >= 100 and max split-R-hat (hyper) <= 1.1;
%   otherwise the arm is INCONCLUSIVE (the ESS-based SE is not reliable).
%   PASS if conclusive AND all 10 hyper |z| <= 3.5 (P(max of 10 |N(0,1)| > 3.5) ~ 0.005) AND voxel
%   frac(|z| > 1.96) <= 0.10 AND voxel max|z| <= 4.5.
%   T8.1A PASSES if the 'shift' arm PASSES. The 'noshift' arm is reported (it is expected to be slow
%   and may be inconclusive).
% Gain (reported, not gated): ESS per 1000 iterations and per second of mu, Sigma (min and median over
%   entries) and of all voxels (median), max R-hat, and the wall time per iteration of each arm.
%
% Part B (K = 2, arm-vs-arm comparison, no exact reference)
%   Same A, s and hyperprior per group; truth: two groups, mu_1 = [-1 0], mu_2 = [1.5 0.5] (B1),
%   mu_1 = [-1.5 0], mu_2 = [1.5 0.5] (B2),
%   Sigma_k = [0.04 0.01; 0.01 0.03], pi = [0.6 0.4], n = 2000. Arms as in A with prior.hierarchical.K = 2.
%   Statistics: mc_compare-type z between the two arms (SE of both arms from their ESS), ordered
%   groups: mean and SD of mu (4), Sigma (6 unique entries), pi_1 (1) -> 22 z; voxels 1..100 u1, u2 ->
%   400 z. Criterion (stated before running): conclusive only if BOTH arms have min hyper ESS >= 100
%   and max hyper R-hat <= 1.1; PASS if conclusive AND all 22 hyper |z| <= 3.5 AND voxel
%   frac(|z| > 1.96) <= 0.10 AND max|z| <= 4.5. Gains reported as in A.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T8_1_shift_move
%
% Notes (first run, 28 Sep 2026, design 1 only): A1 shift arm agreed with the reference on all 410 z
%   (hyper max|z| 1.65, voxels frac 0.025, max 3.22) and mu1 ESS rose 58 -> 1721, but Sigma11 (posterior
%   near 0, u1 poorly measured) mixed slowly in BOTH arms (ESS 31-42, R-hat 1.08): INCONCLUSIVE by the
%   criterion. B1: both arms stuck in different membership configurations (voxel R-hat ~9.5 in both, hyper
%   ESS 4-19): INCONCLUSIVE; the groups overlap in data space along the poorly measured u1.
% Notes (design 2): A2 INCONCLUSIVE again. Sigma mixes slowly in both arms (min ESS 37-46, R-hat 1.09);
%   mu ESS 885/764 -> 9711/9709 with the move, all hyper |z| <= 2.70 in both arms, but voxel z exceed the
%   bound in both arms (frac 0.16 noshift, 0.30 shift; max 4.1 / 4.7), consistent with SEs that ignore the
%   slow Sigma component (voxel ESS ~7500 while Sigma ESS ~40). B2 INCONCLUSIVE (min hyper ESS 6 / 14; mu_k
%   ESS 10-425 -> 87-5955, pi ESS 34 -> 2877, max R-hat 1.78 -> 1.21). The Sigma bottleneck (funnel-type,
%   not a location problem) is not addressed by the shift move.
%
% Kwok-Shing Chan @ MGH
% Date created: 28 September 2026
%

clearvars; tStart = tic;
runK2 = true;

% ---------------------------------------------------------------- settings
d = 2; m = 4; Nv = 2000; ratio = 4; Nchain = 4; Nsub = 100;
muT  = [0.5; -0.3];  SigT = [0.04 0.012; 0.012 0.03];
hp   = struct('m0', [0;0], 'kappa0', 1e-3, 'nu0', 4, 'Psi0', 0.05*eye(2));
zHyperMax = 3.5; zFracMax = 0.10; zMaxMax = 4.5; essMin = 100; rhatMax = 1.1;
seedA = 881; seedData = 882; seedRef = 883; seedRun = 884; seedDataK2 = 885; seedRunK2 = 886;
iter = 40000; burn = 10000; thin = 10;
fprintf('T8.1 joint location-shift move. Seeds: A %d, data %d, ref %d, run %d, K2 data %d, K2 run %d\n', ...
    seedA, seedData, seedRef, seedRun, seedDataK2, seedRunK2);
fprintf('Criterion per arm: conclusive if min hyper ESS >= %d and max hyper R-hat <= %.2f; PASS if all hyper |z| <= %.1f, voxel frac(|z|>1.96) <= %.2f, voxel max|z| <= %.1f\n\n', ...
    essMin, rhatMax, zHyperMax, zFracMax, zMaxMax);

parts = strsplit(getenv('MCMC_BAYES_T8_PARTS'), ',');
if isempty(parts{1}); parts = {'A1','B1','A2','B2'}; end
rng(seedA); A1 = randn(m, d);
A2 = [1 0; 0 1; 0.7 0.7; 0.7 -0.7];
design = struct('A', {A1, A2}, 'ratio', {4, 3}, 'muK', {[-1 1.5; 0 0.5], [-1.5 1.5; 0 0.5]});
verdict = {};
for dz = 1:2
if ~any(strcmp(parts, sprintf('A%d', dz))) && ~any(strcmp(parts, sprintf('B%d', dz))); continue; end
A = design(dz).A; ratio = design(dz).ratio;
fprintf('\n################ design %d ################\n', dz);
VL   = inv(A.'*A);                                          % likelihood covariance / s^2
s    = ratio * mean(sqrt(diag(SigT))) / mean(sqrt(diag(VL)));
fprintf('A = %s, s = %.3f\n', mat2str(A, 3), s);
fprintf('prior SD %s, per-voxel likelihood SD %s (ratio %.1f)\n', mat2str(sqrt(diag(SigT)).', 3), ...
    mat2str(s*sqrt(diag(VL)).', 3), mean(s*sqrt(diag(VL)))/mean(sqrt(diag(SigT))));
fwd = @(p) lingauss_fwd(p, A);

% common fitting
f.modelParams = {'u1';'u2';'noise'}; f.lb = [-Inf; -Inf; 0]; f.ub = [Inf; Inf; 10]; f.xStepSize = [0.2; 0.2; 0.01];
f.algorithm = 'MH'; f.iteration = iter; f.burnin = burn; f.thinning = thin; f.metric = {'mean'};
f.fixedParams = struct('noise', s); f.adaptStepSize = true; f.adaptInterval = 50;
f.repetition = Nchain; f.overdisp = 0.01;
hNIW = struct('hyperprior', 'niw', 'm0', hp.m0, 'kappa0', hp.kappa0, 'nu0', hp.nu0, 'Psi0', hp.Psi0);
mask = true(Nv, 1); x0 = struct('u1', zeros(Nv,1), 'u2', zeros(Nv,1)); iSub = 1:Nsub;

%% ================================================================ Part A (K = 1)
if any(strcmp(parts, sprintf('A%d', dz)))
rng(seedData);
u  = muT + chol(SigT, 'lower')*randn(d, Nv);
y  = A*u + s*randn(m, Nv);                                  % [m, Nv]

% ---------------------------------------------------------------- reference
tRef = tic;
[refTh, refInfo] = reference_collapsed(y, A, s, hp, seedRef);
fprintf('\nReference (collapsed MH, double): %d chains x %d kept (thin 10), acceptance %.3f, %.0f s\n', ...
    size(refTh,3), size(refTh,2), refInfo.acc, toc(tRef));
% hyper quantities [5, Ns, Nch]: mu1 mu2 S11 S21 S22
refQ = theta2q(refTh);
refEss = mcmc_bayes.ess(refQ); refRhat = mcmc_bayes.rhat(refQ);
fprintf('Reference ESS %s, R-hat %s\n', mat2str(round(refEss.'), 6), mat2str(refRhat.', 4));
if any(refRhat > 1.01) || any(refEss < 20000)
    error('T8_1:reference', 'reference not converged / too short (R-hat %s, ESS %s)', mat2str(refRhat.',4), mat2str(round(refEss.')));
end
[refM, refS, refSeM, refSeS] = moments_se(refQ);
qNames = {'mu1','mu2','S11','S21','S22'};
% voxel reference, Rao-Blackwellised over thinned reference draws
[vRefM, vRefS, vRefSeM] = voxel_reference(refTh, y(:, iSub), A, s);
fprintf('reference posterior: mean %s, SD %s\n', mat2str(refM.', 4), mat2str(refS.', 3));

% ---------------------------------------------------------------- arms
yy = reshape(y.', [Nv 1 1 m]);
arms = {'noshift', 'shift'};
resA = struct();
for ka = 1:2
    g = f; g.prior.hierarchical = hNIW;
    if ka == 2; g.prior.hierarchical.shiftMove = true; end
    rng(seedRun); parallel.gpu.rng(seedRun);
    t0 = tic;
    evalc('out = mcmc_bayes().optimisation(yy, mask, [], x0, g, fwd);');
    tRun = toc(t0);
    resA.(arms{ka}) = summarise_arm(out, tRun, g, refM, refS, refSeM, refSeS, iSub, vRefM, vRefS, vRefSeM, ...
        essMin, rhatMax, zHyperMax, zFracMax, zMaxMax, arms{ka}, qNames);
end
print_gain(resA, 'A (K = 1)');
if resA.shift.pass; vA = 'PASS'; elseif ~resA.shift.conclusive; vA = 'INCONCLUSIVE'; else; vA = 'FAIL'; end
verdict{end+1} = sprintf('T8.1A design %d (K = 1, shift arm vs exact reference): %s', dz, vA);
fprintf('\n%s\n', verdict{end});
end

%% ================================================================ Part B (K = 2)
if runK2 && any(strcmp(parts, sprintf('B%d', dz)))
    muK = design(dz).muK; SigK = cat(3, [0.04 0.01; 0.01 0.03], [0.04 0.01; 0.01 0.03]); piK = [0.6; 0.4];
    rng(seedDataK2);
    zT = 1 + (rand(1, Nv) > piK(1));
    uK = zeros(d, Nv);
    for k = 1:2; uK(:, zT==k) = muK(:,k) + chol(SigK(:,:,k), 'lower')*randn(d, nnz(zT==k)); end
    yK = A*uK + s*randn(m, Nv);
    yyK = reshape(yK.', [Nv 1 1 m]);
    outB = cell(1, 2); tB = zeros(1, 2);
    for ka = 1:2
        g = f; g.prior.hierarchical = hNIW; g.prior.hierarchical.K = 2;
        if ka == 2; g.prior.hierarchical.shiftMove = true; end
        rng(seedRunK2); parallel.gpu.rng(seedRunK2);
        t0 = tic;
        evalc('outB{ka} = mcmc_bayes().optimisation(yyK, mask, [], x0, g, fwd);');
        tB(ka) = toc(t0);
    end
    passB = compare_K2(outB, tB, f, iSub, essMin, rhatMax, zHyperMax, zFracMax, zMaxMax);
    verdict{end+1} = sprintf('T8.1B design %d (K = 2, shift vs noshift arm): %s', dz, passB);
    fprintf('\n%s\n', verdict{end});
end

end
fprintf('\n'); fprintf('%s\n', verdict{:});
fprintf('\nTotal time %.1f min\n', toc(tStart)/60);

%% ================================================================ local functions
function s = pf(tf)
if tf; s = 'PASS'; else; s = 'FAIL'; end
end

% hyper quantities [5, Ns, Nch] from the reference parameterisation [5, Ns, Nch] (mu1 mu2 l11 l21 l22)
function q = theta2q(th)
L11 = exp(th(3,:,:)); L21 = th(4,:,:); L22 = exp(th(5,:,:));
q   = cat(1, th(1,:,:), th(2,:,:), L11.^2, L11.*L21, L21.^2 + L22.^2);
end

% mean, SD and their MC standard errors from the multi-chain ESS; x [Nq, Ns, Nch]
function [mn, sd, seM, seS] = moments_se(x)
x   = double(x); Nq = size(x, 1);
xf  = reshape(x, Nq, []);
mn  = mean(xf, 2); sd = std(xf, 0, 2);
seM = sd ./ sqrt(mcmc_bayes.ess(x));
d2  = (x - mn).^2;
seV = sqrt(var(reshape(d2, Nq, []), 0, 2) ./ mcmc_bayes.ess(d2));
seS = seV ./ (2*sd);
end

% collapsed posterior of (mu, Sigma), log density in theta = [mu; log L11; L21; log L22] incl. Jacobian
function lp = logpost_collapsed(th, ybar, Syy, n, A, s, hp)
m   = size(A, 1);
L   = [exp(th(3)) 0; th(4) exp(th(5))];
Sig = L*L.';
M   = A*Sig*A.' + s^2*eye(m);
[R, flag] = chol(M);
if flag; lp = -Inf; return; end
r   = ybar - A*th(1:2);
T   = Syy + n*(r*r.');
Ri  = R \ eye(m);
ll  = -n*sum(log(diag(R))) - 0.5*sum(sum((Ri*Ri.') .* T));
Li  = L \ eye(2); Sinv = Li.'*Li;
logdetS = 2*(th(3) + th(5));
a   = Li*(th(1:2) - hp.m0);
lpr = -0.5*logdetS - 0.5*hp.kappa0*(a.'*a) - (hp.nu0 + 2 + 1)/2*logdetS - 0.5*sum(sum(hp.Psi0 .* Sinv));
lp  = ll + lpr + 3*th(3) + 2*th(5);
end

% long random-walk MH on the collapsed posterior (host, double)
function [thAll, info] = reference_collapsed(y, A, s, hp, seed)
rng(seed);
[m, n] = size(y);
ybar = mean(y, 2); yc = y - ybar; Syy = yc*yc.';
lpf  = @(th) logpost_collapsed(th, ybar, Syy, n, A, s, hp);
% start: method of moments
B    = (A.'*A) \ A.';
uh   = B*y;
Sig0 = cov(uh.') - s^2*inv(A.'*A);
[V, D] = eig((Sig0 + Sig0.')/2); Sig0 = V*diag(max(diag(D), 1e-3))*V.';
L0   = chol(Sig0, 'lower');
th   = [mean(uh, 2); log(L0(1,1)); L0(2,1); log(L0(2,2))];
% pilot: 4 x 20000 with covariance refresh
C = diag([1e-4 1e-4 0.01 0.01 0.01]);
for pilot = 1:4
    [ch, ~] = rw_chain(lpf, th, C, 20000);
    th = ch(:, end);
    C  = cov(ch(:, 5001:end).') * 2.38^2/5 + 1e-12*eye(5);
end
% main: 4 chains, over-dispersed starts from the pilot, burn-in 20000, 250000 kept (thin 10)
Nch = 4; Nkeep = 250000; thinR = 10;
thAll = zeros(5, Nkeep/thinR, Nch); accs = zeros(1, Nch);
Lc = chol(C/(2.38^2/5), 'lower');
for c = 1:Nch
    th0 = th + 3*Lc*randn(5, 1);
    [ch, ~] = rw_chain(lpf, th0, C, 20000);
    [ch, accs(c)] = rw_chain(lpf, ch(:, end), C, Nkeep);
    thAll(:,:,c) = ch(:, thinR:thinR:end);
end
info.acc = mean(accs); info.C = C;
end

function [ch, acc] = rw_chain(lpf, th, C, N)
L  = chol(C, 'lower');
lp = lpf(th);
ch = zeros(numel(th), N); nacc = 0;
Z  = randn(numel(th), N); U = log(rand(1, N));
for k = 1:N
    thp = th + L*Z(:,k);
    lpp = lpf(thp);
    if U(k) < lpp - lp
        th = thp; lp = lpp; nacc = nacc + 1;
    end
    ch(:, k) = th;
end
acc = nacc / N;
end

% Rao-Blackwellised voxel posterior mean/SD (and SE of the mean) over the reference draws
function [mRef, sRef, seRef] = voxel_reference(thAll, ySub, A, s)
[~, Ns, Nch] = size(thAll);
th  = reshape(thAll, 5, []);
Nd  = size(th, 2); Nsub = size(ySub, 2);
mi  = zeros(2*Nsub, Nd); Cpp = zeros(2, Nd);
Aty = A.'*ySub/s^2; AtA = A.'*A/s^2;
for j = 1:Nd
    L   = [exp(th(3,j)) 0; th(4,j) exp(th(5,j))];
    P   = (L*L.') \ eye(2);
    C   = inv(AtA + P);
    mm  = C*(Aty + P*th(1:2,j));
    mi(:, j) = mm(:);
    Cpp(:, j) = diag(C);
end
mRef  = mean(mi, 2);                                        % [2*Nsub,1], order (u1,u2) per voxel
vRef  = repmat(mean(Cpp, 2), Nsub, 1) + var(mi, 0, 2);
sRef  = sqrt(vRef);
seRef = std(mi, 0, 2) ./ sqrt(mcmc_bayes.ess(reshape(mi, 2*Nsub, Ns, Nch)));
% reorder to [u1 of voxels; u2 of voxels]
ord   = [1:2:2*Nsub, 2:2:2*Nsub];
mRef  = mRef(ord); sRef = sRef(ord); seRef = seRef(ord);
end

% hyper samples [5, Ns, Nrep] (mu1 mu2 S11 S21 S22) from out.hyper (K = 1)
function q = hyper_q(out)
mu = out.hyper.posterior.mu;  S = out.hyper.posterior.Sigma;
q  = cat(1, mu(1,:,:), mu(2,:,:), reshape(S(1,1,:,:), 1, size(S,3), []), ...
         reshape(S(2,1,:,:), 1, size(S,3), []), reshape(S(2,2,:,:), 1, size(S,3), []));
end

function r = summarise_arm(out, tRun, g, refM, refS, refSeM, refSeS, iSub, vRefM, vRefS, vRefSeM, ...
                           essMin, rhatMax, zHyperMax, zFracMax, zMaxMax, name, qNames)
q = double(hyper_q(out));
[mn, sd, seM, seS] = moments_se(q);
zH   = [(mn - refM) ./ sqrt(seM.^2 + refSeM.^2); (sd - refS) ./ sqrt(seS.^2 + refSeS.^2)];
essH = mcmc_bayes.ess(q); rhH = mcmc_bayes.rhat(q);
% voxels 1..Nsub, [u1; u2]
xv   = double(cat(1, out.posterior.u1(iSub,:,:), out.posterior.u2(iSub,:,:)));
[vm, vs, vseM, vseS] = moments_se(xv);
zV   = [(vm - vRefM) ./ sqrt(vseM.^2 + vRefSeM.^2); (vs - vRefS) ./ vseS];
Nit  = g.iteration * g.repetition;
essV = [out.diagnostics.ess.u1(:); out.diagnostics.ess.u2(:)];
rhV  = [out.diagnostics.rhat.u1(:); out.diagnostics.rhat.u2(:)];
r.conclusive = min(essH) >= essMin && max(rhH) <= rhatMax;
r.passH  = all(abs(zH) <= zHyperMax);
r.fracV  = mean(abs(zV) > 1.96); r.maxV = max(abs(zV));
r.pass   = r.conclusive && r.passH && r.fracV <= zFracMax && r.maxV <= zMaxMax;
r.essH = essH; r.rhH = rhH; r.zH = zH; r.essV = essV; r.rhV = rhV;
r.tIter = tRun / Nit; r.tRun = tRun; r.Nit = Nit;
r.essHper1k = essH / Nit * 1000; r.essVper1k = median(essV) / Nit * 1000;
fprintf('\n--- arm %s: %.1f s, %.3f ms/iteration\n', name, tRun, 1e3*r.tIter);
if isfield(out.diagnostics, 'shiftMove')
    sm = out.diagnostics.shiftMove;
    fprintf('    shift acceptance (kept) %s, burn-in %s, frozen scale %s\n', mat2str(sm.acceptance, 3), ...
        mat2str(sm.acceptanceBurnin, 3), mat2str(sm.scale, 3));
end
fprintf('    %-5s %9s %9s %9s %9s %8s %8s %7s %7s\n', 'q', 'mean', 'ref', 'sd', 'refSD', 'ESS', 'R-hat', 'z_mean', 'z_sd');
for k = 1:numel(qNames)
    fprintf('    %-5s %9.5f %9.5f %9.5f %9.5f %8.0f %8.4f %7.2f %7.2f\n', qNames{k}, mn(k), refM(k), sd(k), refS(k), ...
        essH(k), rhH(k), zH(k), zH(k+numel(qNames)));
end
fprintf('    voxels (all %d): median ESS %.0f, max R-hat %.4f; subset z: frac(|z|>1.96) %.3f, max|z| %.2f\n', ...
    numel(essV)/2, median(essV), max(rhV), r.fracV, r.maxV);
if ~r.conclusive
    fprintf('    arm %s: INCONCLUSIVE (min hyper ESS %.0f, max hyper R-hat %.3f)\n', name, min(essH), max(rhH));
else
    fprintf('    arm %s: %s (hyper max|z| %.2f)\n', name, pf(r.pass), max(abs(zH)));
end
end

function print_gain(res, label)
a = res.noshift; b = res.shift;
fprintf('\nGain %s (shift / noshift):\n', label);
fprintf('    hyper ESS per 1000 it: noshift min %.1f med %.1f | shift min %.1f med %.1f | ratio min %.1f med %.1f\n', ...
    min(a.essHper1k), median(a.essHper1k), min(b.essHper1k), median(b.essHper1k), ...
    min(b.essHper1k)/min(a.essHper1k), median(b.essHper1k)/median(a.essHper1k));
fprintf('    hyper ESS per second : ratio min %.1f\n', (min(b.essH)/b.tRun)/(min(a.essH)/a.tRun));
fprintf('    hyper max R-hat      : noshift %.4f | shift %.4f\n', max(a.rhH), max(b.rhH));
fprintf('    voxel median ESS per 1000 it: noshift %.1f | shift %.1f | ratio %.2f; voxel max R-hat %.4f | %.4f\n', ...
    a.essVper1k, b.essVper1k, b.essVper1k/a.essVper1k, max(a.rhV), max(b.rhV));
fprintf('    time per iteration   : noshift %.3f ms | shift %.3f ms (x%.2f)\n', 1e3*a.tIter, 1e3*b.tIter, b.tIter/a.tIter);
end

% K = 2: arm-vs-arm comparison (ordered groups)
function res = compare_K2(outB, tB, f, iSub, essMin, rhatMax, zHyperMax, zFracMax, zMaxMax)
names = {'noshift','shift'};
Q = cell(1,2); essH = cell(1,2); rhH = cell(1,2);
for ka = 1:2
    H  = outB{ka}.hyper.posterior;                          % mu [d,K,Ns,Nrep], Sigma [d,d,K,Ns,Nrep], pi [K,Ns,Nrep]
    Ns = size(H.pi, 2); Nr = size(H.pi, 3);
    mu = reshape(H.mu, 4, Ns, Nr);
    S  = reshape(H.Sigma, 8, Ns, Nr); S = S([1 2 4 5 6 8], :, :);    % S11 S21 S22 per group
    Q{ka} = double(cat(1, mu, S, H.pi(1,:,:)));
    essH{ka} = mcmc_bayes.ess(Q{ka}); rhH{ka} = mcmc_bayes.rhat(Q{ka});
end
qn = {'mu1_1','mu2_1','mu1_2','mu2_2','S11_1','S21_1','S22_1','S11_2','S21_2','S22_2','pi_1'};
[ma, sa, seMa, seSa] = moments_se(Q{1}); [mb, sb, seMb, seSb] = moments_se(Q{2});
zH = [(mb - ma) ./ sqrt(seMa.^2 + seMb.^2); (sb - sa) ./ sqrt(seSa.^2 + seSb.^2)];
fprintf('\n--- Part B (K = 2): noshift %.1f s (%.3f ms/it), shift %.1f s (%.3f ms/it, x%.2f)\n', tB(1), ...
    1e3*tB(1)/(f.iteration*f.repetition), tB(2), 1e3*tB(2)/(f.iteration*f.repetition), tB(2)/tB(1));
sm = outB{2}.diagnostics.shiftMove;
fprintf('    shift acceptance (kept, internal labels) %s, frozen scale %s\n', mat2str(sm.acceptance, 3), mat2str(sm.scale, 3));
fprintf('    %-6s %9s %9s %8s %8s %8s %8s %7s %7s\n', 'q', 'mean_ns', 'mean_s', 'ESS_ns', 'ESS_s', 'Rh_ns', 'Rh_s', 'z_mean', 'z_sd');
for k = 1:numel(qn)
    fprintf('    %-6s %9.5f %9.5f %8.0f %8.0f %8.4f %8.4f %7.2f %7.2f\n', qn{k}, ma(k), mb(k), essH{1}(k), essH{2}(k), ...
        rhH{1}(k), rhH{2}(k), zH(k), zH(k+numel(qn)));
end
xa = double(cat(1, outB{1}.posterior.u1(iSub,:,:), outB{1}.posterior.u2(iSub,:,:)));
xb = double(cat(1, outB{2}.posterior.u1(iSub,:,:), outB{2}.posterior.u2(iSub,:,:)));
[zVm, zVs] = mc_compare(xb, xa); zV = [zVm; zVs];
Nit = f.iteration * f.repetition;
for ka = 1:2
    essV = [outB{ka}.diagnostics.ess.u1(:); outB{ka}.diagnostics.ess.u2(:)];
    rhV  = [outB{ka}.diagnostics.rhat.u1(:); outB{ka}.diagnostics.rhat.u2(:)];
    fprintf('    %-7s hyper ESS/1000 it min %.1f med %.1f, max R-hat %.4f; voxel median ESS/1000 it %.1f, max R-hat %.4f\n', ...
        names{ka}, min(essH{ka})/Nit*1000, median(essH{ka})/Nit*1000, max(rhH{ka}), median(essV)/Nit*1000, max(rhV));
end
conclusive = min(essH{1}) >= essMin && min(essH{2}) >= essMin && max(rhH{1}) <= rhatMax && max(rhH{2}) <= rhatMax;
fracV = mean(abs(zV) > 1.96); maxV = max(abs(zV));
fprintf('    hyper max|z| %.2f; voxel frac(|z|>1.96) %.3f, max|z| %.2f\n', max(abs(zH)), fracV, maxV);
if ~conclusive
    res = sprintf('INCONCLUSIVE (min hyper ESS noshift %.0f / shift %.0f, max R-hat %.3f / %.3f)', ...
        min(essH{1}), min(essH{2}), max(rhH{1}), max(rhH{2}));
else
    res = pf(all(abs(zH) <= zHyperMax) && fracV <= zFracMax && maxV <= zMaxMax);
end
end
