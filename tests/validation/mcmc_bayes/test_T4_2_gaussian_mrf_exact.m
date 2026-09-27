%% test_T4_2_gaussian_mrf_exact.m
%
% T4.2 (Phase 4): exact Gaussian-MRF test of the chromatic sweep.
%
% Model (gmrf_exact_run.m): linear Gaussian toy y_i = A u_i + e_i, known s = 1, d = 2, m = 4,
%   8 x 8 x 4 grid with a mask (~15% holes), fixed mu, Sigma (SDs 0.8, 0.5, corr 0.3),
%   quadratic potential rho = x^2/2, W = 1./sqrt(diag(Sigma)), fixed symmetric edge weights
%   w_ij in [0.5, 1.5]. The joint posterior is Gaussian with precision
%       Q = I (x) (A'A/s^2 + Sigma^-1) + (1/tau) L_w (x) diag(W)
%   (L_w the weighted graph Laplacian, built by brute force; each edge counted once with
%   rho = x^2/2 gives exactly (1/tau) L_w (x) diag(W), see the mcmc_bayes header; checked
%   numerically: |sum of the sampler's local differences over a colour - dPhi from Q| / |dPhi|
%   <= 1e-10). Mean Q\b and covariance inv(Q) in double on the host.
%
% Configurations (gmrf_exact_suite.m). Coupling = mean diagonal of the MRF precision relative
%   to the local precision A'A/s^2 + Sigma^-1 (6.1): ~0.25 x "moderate", ~2.3 x "strong",
%   ~11 x "very strong" (printed per configuration):
%   3d face (K = 6, 2 colours): tau = 5, 0.5, 0.1 joint; tau = 0.5 componentwise
%   2d full r = 2 (K = 24, 9 colours): tau = 15, 1.5, 0.3 joint
%   4 chains (stacked copies), 30000 iterations, burn-in 6000, thinning 5 (4800 draws/chain),
%   adaptStepSize (burn-in only).
%
% Test statistics (per configuration), z = (sampler mean of t - exact E[t]) / (sd(t)/sqrt(ESS(t))):
%   means (t = u_ip), marginal variances (t = (u_ip - m_ip)^2), within-voxel covariance
%   (t = (u_i1 - m_i1)(u_i2 - m_i2)), neighbour cross-covariances for every edge and both
%   parameters (t = (u_ip - m_ip)(u_jp - m_jp)).
% Criterion (stated before running), per configuration:
%   frac(|z| > 1.96) <= 0.10  AND  max|z| <= z*(N) = Bonferroni 1% two-sided over the N statistics
%   (z* = sqrt(2) erfcinv(0.01/N), ~4.6-4.9 here), AND discrimination: median |z| of the sampler
%   means against the exact means WITHOUT the MRF >= 3, AND the scaling check above.
%   T4.2 PASS if all configurations pass. R-hat, ESS, acceptance and ratios (sampler/exact)
%   of variances and neighbour covariances are reported.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T4_2_gaussian_mrf_exact
% Environment: MCMC_BAYES_T4_ITER overrides the number of iterations (pilot runs; burn-in 20%).
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%

clearvars; tStart = tic;

iteration = 30000;
if ~isempty(getenv('MCMC_BAYES_T4_ITER')); iteration = str2double(getenv('MCMC_BAYES_T4_ITER')); end
[isPass, labels] = gmrf_exact_suite('chromatic', iteration);
fprintf('T4.2 overall: %s   (total time %.1f min)\n', pf(all(isPass)), toc(tStart)/60);

function s = pf(c)
if c; s = 'PASS'; else; s = 'FAIL'; end
end
