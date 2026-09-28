%% test_T3_4_broad_prior_regression.m
%
% T3.4 (Phase 3): regression of the hierarchical Normal prior against the
% Phase 1/2 flat-prior path, with a very broad FIXED prior on u.
%
% What is expected. The hierarchical prior N(u | mu, Sigma) is a density in the
% transformed space u = T(x), and the hierarchical parameters get no
% log-Jacobian. With Sigma = 1e6 I (SD 1000 in u) the prior is flat in u over
% the posterior region (log-prior variation < 1e-3 there), so the target is
%       L(x(u)) * const      (flat in u),
% whereas the Phase 1/2 path with a non-linear transform targets
%       L(x(u)) * |dx/du|    (flat in native x).
% The two differ by the Jacobian, so the meaningful regression is against a
% Phase 1/2 run that is ALSO flat in u: a reparameterised model whose sampled
% parameter is u itself ('linear' transform, box by rejection), with the forward
% model mapping u -> x. For R2star with the 'log' transform this is
% uR2 = log(R2star) in the box [log 0.1, log 200]; at SNR 20-50 the posterior
% is far inside the box, so the box has no effect. For 'linear' parameters
% (M0 here) flat-in-u is flat-in-x and the Phase 1 native box [0, 2] is used.
% (Why not IVIM: flat-in-u priors on logit F or log Dstar give improper posteriors
% when F -> 0 or Dstar -> Inf leaves a non-zero likelihood; the monoexponential
% likelihood vanishes fast enough in both tails of log R2star.)
%
% Cases (monoexponential, gpuR2starMapping, 100 voxels, r2star_sim.m, SNR 20 and 50)
%   (a) likelihood 'gaussian', noise sampled with its box (not under the
%       hierarchy, Phase 1 treatment). Hierarchical: M0 'linear' (-Inf, Inf),
%       R2star 'log' (0, Inf), mu = [0 0], Sigma = 1e6 I, fixed.
%       Reference: modelParams {M0, uR2, noise}, all 'linear', flat in u.
%   (b) likelihood 'marginal_S0noise' (S0Param M0). Hierarchical: R2star 'log'
%       (0, Inf) only. Reference: {M0, uR2, noise} with uR2 'linear'.
%   (c) information only: (a)'s hierarchical run vs the Phase 1 flat-NATIVE run
%       ('log' transform on R2star with the Jacobian): the expected difference
%       (posterior tilted by 1/R2star) shows the comparison can detect a
%       Jacobian error.
% Sampler: joint, adaptStepSize (burn-in only), 4 chains per voxel (run_chains.m;
%   fixed mode, so voxel copies are independent), starts M0 ~ U(0.9,1.1),
%   R2star ~ U(15,60), noise ~ U(0.01,0.08) per chain copy. 40000 iterations,
%   burn-in 10000, thinning 5.
% Test statistic: mc_compare.m z-scores of the posterior mean and SD difference
%   between the two runs, per voxel and output parameter (a: M0, R2star, noise;
%   b: R2star, noise, M0 (recovered)).
% Criterion (stated before running), per case pooled over SNR, parameters and
%   mean/SD (100 x 3 x 2 x 2 = 1200 z): PASS if frac(|z| > 1.96) <= 0.10 AND
%   max|z| <= 4.5.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T3_4_broad_prior_regression
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%

clearvars; tStart = tic;

Nv = 100; SNRs = [20 50]; Nchain = 4;
SigmaBroad = 1e6;
zFracMax = 0.10; zMaxMax = 4.5;
seedData = [91 92]; seedStart = 93; seedRun = [94 95 96 97 98];

fprintf('T3.4 broad fixed prior (Sigma = %g I in u) vs flat-in-u Phase 1/2 reference: Nv=%d, SNR %s, %d chains\n', SigmaBroad, Nv, mat2str(SNRs), Nchain);
fprintf('Seeds: data %s, start %d, runs %s\n', mat2str(seedData), seedStart, mat2str(seedRun));
fprintf('Criterion per case: frac(|z|>1.96) <= %.2f and max|z| <= %.1f (pooled over SNR, parameters, mean/SD)\n\n', zFracMax, zMaxMax);

zA = []; zB = []; zC = [];
for ks = 1:numel(SNRs)
    SNR = SNRs(ks);
    [y4, ~, ~, fit0, obj] = r2star_sim(Nv, SNR, seedData(ks));
    y   = double(reshape(y4, Nv, []));
    fwd = @(p) obj.FWD(p, 'mcmc');
    fwdU = @(p) obj.FWD(struct('M0', p.M0, 'R2star', exp(p.uR2), 'noise', p.noise), 'mcmc');
    iN  = strcmp(fit0.modelParams, 'noise');
    lbN = fit0.lb(iN); ubN = fit0.ub(iN);

    base = struct('algorithm','MH', 'iteration',40000, 'burnin',10000, 'thinning',5, 'metric',{{'mean'}}, ...
                  'updateScheme','joint', 'adaptStepSize',true, 'adaptInterval',50);
    rng(seedStart + ks);
    N0  = Nv*Nchain;
    st  = struct('M0', 0.9 + 0.2*rand(N0,1), 'R2star', 15 + 45*rand(N0,1), 'noise', 0.01 + 0.07*rand(N0,1));
    stU = struct('M0', st.M0, 'uR2', log(st.R2star), 'noise', st.noise);

    % ---------- (a) gaussian ----------
    fH = base; fH.modelParams = {'M0';'R2star';'noise'};
    fH.lb = [-Inf; 0; lbN]; fH.ub = [Inf; Inf; ubN]; fH.xStepSize = [0.01; 0.5; 0.002];
    fH.parameterTransform = {'linear','log','linear'};
    fH.prior.hierarchical = struct('fixed', true, 'mu', [0 0], 'Sigma', SigmaBroad*eye(2));
    [pH, ~, tH] = run_chains(y, fH, fwd, st, Nchain, seedRun(1));

    fR = base; fR.modelParams = {'M0';'uR2';'noise'};
    fR.lb = [0; log(0.1); lbN]; fR.ub = [2; log(200); ubN]; fR.xStepSize = [0.01; 0.02; 0.002];
    fR.parameterTransform = 'linear'; fR.forceNewPath = true;
    [pR, ~, tR] = run_chains(y, fR, fwdU, stU, Nchain, seedRun(2));
    pR.R2star = exp(pR.uR2);

    za = compare_runs(pH, pR, {'M0','R2star','noise'});
    zA = [zA; za]; %#ok<AGROW>
    fprintf('SNR %d (a) gaussian: runs %.0f/%.0f s | frac|z|>1.96 %.3f, max|z| %.2f (n=%d)\n', SNR, tH, tR, mean(abs(za)>1.96), max(abs(za)), numel(za));

    % ---------- (c) information: flat-native Phase 1 run ('log' with the Jacobian) ----------
    fN = base; fN.modelParams = {'M0';'R2star';'noise'};
    fN.lb = [0; 0.1; lbN]; fN.ub = [2; 200; ubN]; fN.xStepSize = [0.01; 0.5; 0.002];
    fN.parameterTransform = {'linear','log','linear'};
    [pN, ~, tN] = run_chains(y, fN, fwd, st, Nchain, seedRun(3));
    [zMc, zSc] = mc_compare(pH.R2star, pN.R2star);
    zC = [zC; zMc]; %#ok<AGROW>
    fprintf('SNR %d (c) info, hierarchical (flat in log R2star) vs flat native R2star: run %.0f s | R2star mean z: median %+.2f, frac|z|>1.96 %.2f; SD z median %+.2f\n', ...
        SNR, tN, median(zMc), mean(abs(zMc)>1.96), median(zSc));

    % ---------- (b) marginal_S0noise ----------
    gH = base; gH.modelParams = {'M0';'R2star';'noise'};
    gH.lb = [0; 0; lbN]; gH.ub = [2; Inf; ubN]; gH.xStepSize = [0.01; 0.5; 0.002];
    gH.parameterTransform = {'linear','log','linear'};
    gH.likelihood = 'marginal_S0noise'; gH.S0Param = 'M0';
    gH.prior.hierarchical = struct('fixed', true, 'mu', 0, 'Sigma', SigmaBroad);
    [qH, ~, tH] = run_chains(y, gH, fwd, st, Nchain, seedRun(4));

    gR = base; gR.modelParams = {'M0';'uR2';'noise'};
    gR.lb = [0; log(0.1); lbN]; gR.ub = [2; log(200); ubN]; gR.xStepSize = [0.01; 0.02; 0.002];
    gR.parameterTransform = 'linear';
    gR.likelihood = 'marginal_S0noise'; gR.S0Param = 'M0';
    [qR, ~, tR] = run_chains(y, gR, fwdU, stU, Nchain, seedRun(5));
    qR.R2star = exp(qR.uR2);

    zb = compare_runs(qH, qR, {'R2star','noise','M0'});
    zB = [zB; zb]; %#ok<AGROW>
    fprintf('SNR %d (b) marginal_S0noise: runs %.0f/%.0f s | frac|z|>1.96 %.3f, max|z| %.2f (n=%d)\n\n', SNR, tH, tR, mean(abs(zb)>1.96), max(abs(zb)), numel(zb));
end

passA = mean(abs(zA) > 1.96) <= zFracMax && max(abs(zA)) <= zMaxMax;
passB = mean(abs(zB) > 1.96) <= zFracMax && max(abs(zB)) <= zMaxMax;
fprintf('(a) gaussian pooled: n=%d, frac|z|>1.96 = %.3f, max|z| = %.2f -> %s\n', numel(zA), mean(abs(zA)>1.96), max(abs(zA)), pf(passA));
fprintf('(b) marginal pooled: n=%d, frac|z|>1.96 = %.3f, max|z| = %.2f -> %s\n', numel(zB), mean(abs(zB)>1.96), max(abs(zB)), pf(passB));
fprintf('(c) information: flat-in-u vs flat-native R2star means, pooled median z %+.2f, frac|z|>1.96 %.2f (expected: clearly non-zero)\n', median(zC), mean(abs(zC)>1.96));
fprintf('\nT3.4 overall: %s   (total time %.1f min)\n', pf(passA && passB), toc(tStart)/60);

%% local functions
function z = compare_runs(pA, pB, params)
% pooled mean and SD z-scores (mc_compare.m) over the listed parameters
z = [];
for k = 1:numel(params)
    [zM, zS] = mc_compare(pA.(params{k}), pB.(params{k}));
    z = [z; zM(:); zS(:)]; %#ok<AGROW>
end
end

function s = pf(c)
if c; s = 'PASS'; else; s = 'FAIL'; end
end
