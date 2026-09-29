%% test_T9c_0_labels_exact.m
%
% T9c.0 (Phase 9c): exactness of the fixed-label prior with fixed hyperparameters (PASS/FAIL).
%
% Linear Gaussian model y_i = A u_i + e, e ~ N(0, s^2 I) (d = 2, m = 4, s = 1, known noise), two label groups
% (label values 3 and 8, to exercise the label -> group mapping) with FIXED (mu_k, Sigma_k). The exact posterior
% of voxel i with group g_i is Gaussian:
%       u_i | y_i ~ N(C (A'y_i/s^2 + P_g mu_g), C),   C = (A'A/s^2 + P_g)^-1,  P_g = Sigma_g^-1.
% 40 voxels, 20 per group; 6 voxels of each group have data generated from the OTHER group, so that the prior
% term (the voxel's own group, not the mixture) matters. Every voxel is replicated in Nch independent chains
% (one GPU run, the replicas are independent voxels of the fixed-mode prior).
%
% Criterion (as T3.1 / T7.1 / T9b.0): for the 5%, 25%, 50%, 75%, 95% quantiles of u1 and u2 of every voxel,
%   z = (pooled sample quantile - exact) / (SD of the per-chain quantiles / sqrt(Nch)), 400 z values:
%   PASS if frac(|z| > 1.96) <= 0.10 and max |z| <= 4.5.
% Sizes: Nch = 64 chains, 12000 iterations, burn-in 3000, thinning 3 (~1 min on an A40). The unit test
%   McmcBayesUnitTest/testLabelsExactToyShort is a short version (12 voxels, 32 chains, 6000 iterations,
%   frac <= 0.15).
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd); addpath(fullfile(pwd,'tests'));
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T9c_0_labels_exact
%
% Kwok-Shing Chan @ MGH
% Date created: 29 September 2026
%

clearvars; tStart = tic;
d = 2; m = 4; s = 1; Nch = 64; Nv = 40;
mu = [-1 1.5; 0 1]; Sig = cat(3, [0.3 0.1; 0.1 0.2], [0.25 -0.05; -0.05 0.4]); lv = [3 8];
rng(9701); A = randn(m, d);
g     = [ones(1, 20), 2*ones(1, 20)];                       % group of the prior (label lv(g))
gData = g; gData([1:6, 21:26]) = 3 - g([1:6, 21:26]);        % 6 voxels per group with data of the other group
u = zeros(d, Nv);
for i = 1:Nv; u(:,i) = mu(:,gData(i)) + chol(Sig(:,:,gData(i)), 'lower')*randn(d, 1); end
y = A*u + s*randn(m, Nv);

% exact marginal quantiles
qs = [0.05 0.25 0.5 0.75 0.95]; zq = -sqrt(2)*erfcinv(2*qs);
qx = zeros(Nv, d, numel(qs));
for i = 1:Nv
    P = inv(Sig(:,:,g(i))); C = inv(A.'*A/s^2 + P); mP = C*(A.'*y(:,i)/s^2 + P*mu(:,g(i)));
    for p = 1:d; qx(i,p,:) = mP(p) + sqrt(C(p,p))*zq; end
end

yy = reshape(repmat(y.', Nch, 1), [Nv*Nch 1 1 m]);
f.modelParams = {'u1';'u2';'noise'}; f.lb = [-Inf; -Inf; 0]; f.ub = [Inf; Inf; 10]; f.xStepSize = [0.3; 0.3; 0.01];
f.algorithm = 'MH'; f.iteration = 12000; f.burnin = 3000; f.thinning = 3; f.metric = {'mean'};
f.fixedParams = struct('noise', s); f.adaptStepSize = true; f.adaptInterval = 50;
f.prior.hierarchical = struct('labels', repmat(lv(g).', Nch, 1), 'fixed', true, 'mu', mu, 'Sigma', Sig);
rng(9702); x0.u1 = 2*randn(Nv*Nch, 1); x0.u2 = 2*randn(Nv*Nch, 1);
rng(9703); parallel.gpu.rng(9703);
[~, out] = evalc('mcmc_bayes().optimisation(yy, true(Nv*Nch, 1), [], x0, f, @(p) McmcBayesUnitTest.linGaussFwd(p, A))');

z = zeros(Nv, d, numel(qs));
for p = 1:d
    U = reshape(double(out.posterior.(sprintf('u%d', p))), Nv, Nch, []);
    for k = 1:numel(qs)
        qc = McmcBayesUnitTest.sampleQuantile(U, qs(k), 3);
        qp = McmcBayesUnitTest.sampleQuantile(reshape(permute(U, [1 3 2]), Nv, []), qs(k), 2);
        z(:,p,k) = (qp - qx(:,p,k)) ./ (std(qc, 0, 2)/sqrt(Nch));
    end
end
fr = mean(abs(z(:)) > 1.96); mz = max(abs(z(:)));
zMis = z([1:6, 21:26], :, :);
fprintf('T9c.0 fixed labels (3, 8), exact Gaussian posterior: %d voxels x %d chains, %d kept samples per chain\n', Nv, Nch, size(out.posterior.u1, 2));
fprintf('  frac(|z| > 1.96) = %.3f (<= 0.10), max |z| = %.2f (<= 4.5); voxels with data of the other group: frac %.3f, max %.2f\n', ...
        fr, mz, mean(abs(zMis(:)) > 1.96), max(abs(zMis(:))));
fprintf('  labels in the output: %s, group sizes %s\n', mat2str(out.hyper.labels), mat2str(out.hyper.groupSize.'));
if fr <= 0.10 && mz <= 4.5; fprintf('T9c.0 PASS'); else; fprintf('T9c.0 FAIL'); end
fprintf(' (%.1f min)\n', toc(tStart)/60);
