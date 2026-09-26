%% test_T1_neutral_vs_legacy.m
%
% Phase 1: the neutral new sampling path agrees with legacy mcmc.
%
% The neutral new path is parameterTransform 'linear', updateScheme
% 'joint', no adaptation, overdisp 0. isLegacy() classifies these options
% as legacy (so mcmc_bayes would pass through to mcmc), therefore the
% test-only switch fitting.forceNewPath = true is used to force the new
% loop (mcmc_bayes.metropolis_hastings_bayes).
%
% Monoexponential R2* model (gpuR2starMapping FWD), 1000 voxels, SNR 20
% and 50, 12 echoes, class default start points and xStepSize.
%
% Criteria (stated before running), for each SNR
%   (a) statistical agreement, independent seeds for new and legacy:
%       - KS: on a random subset of 200 voxels x 3 parameters (chains
%         thinned by 2*tau, thin_by_ess.m), #rejections at alpha = 0.05
%         within the exact central 99% interval of Binomial(600, 0.05);
%       - moments: per voxel/parameter z-scores of the posterior mean and
%         SD difference (mc_compare.m); frac(|z| > 1.96) <= 0.075 and
%         |median z| <= 0.15 per parameter (as in T1.3).
%   (b) information only (not a pass criterion, bitwise identity is not
%       required by the plan): with the SAME seed, is the new path
%       bitwise identical to legacy? The neutral loop draws random numbers
%       in the same order as legacy, so it is expected to be.
%   (c) information only: wall time new/legacy for the same run (one
%       forward evaluation per iteration in both).
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T1_neutral_vs_legacy
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%

clearvars; tStart = tic;

% settings
Nv          = 1000;
Nsub        = 200;
SNRs        = [20 50];
alpha       = 0.05;
seedData    = [20260926 20260927];  % per SNR (same data as T1.2/T1.3)
seedNew     = 3001;                 % new path
seedLegacy  = 3002;                 % legacy, independent run
seedSubset  = 11;
zFracMax    = 0.075;
zMedMax     = 0.15;

iteration   = 30000;
burnin      = 5000;
thinning    = 5;

nTest       = Nsub * 3;
[lo, hi]    = binom_bounds(nTest, alpha, 0.01);

fprintf('T1 neutral vs legacy: Nv=%d, subset=%d, SNR=%s, iteration=%d, burnin=%d, thinning=%d\n', ...
    Nv, Nsub, mat2str(SNRs), iteration, burnin, thinning);
fprintf('Seeds: data %s, new %d, legacy %d (independent) / %d (same-seed check), subset %d\n', ...
    mat2str(seedData), seedNew, seedLegacy, seedNew, seedSubset);
fprintf('Criteria: KS #rejections in [%d, %d] of %d; frac(|z|>1.96) <= %.3f and |median z| <= %.2f (mean and SD, per parameter)\n', ...
    lo, hi, nTest, zFracMax, zMedMax);

isPass = true;
for ksnr = 1:numel(SNRs)
    SNR = SNRs(ksnr);
    [y, mask, pars0, fitting, obj] = r2star_sim(Nv, SNR, seedData(ksnr));
    w = ones(size(y), 'single');
    params = fitting.modelParams;

    fitting.iteration   = iteration;
    fitting.burnin      = burnin;
    fitting.thinning    = thinning;
    fitting.repetition  = 1;
    fNew                = fitting;
    fNew.forceNewPath   = true;

    % new path, legacy with an independent seed, legacy with the same seed
    rng(seedNew); parallel.gpu.rng(seedNew);
    t0 = tic; evalc('outNew = mcmc_bayes().optimisation(y, mask, w, pars0, fNew, @obj.FWD, ''mcmc'', fNew);'); tNew = toc(t0);
    rng(seedLegacy); parallel.gpu.rng(seedLegacy);
    t0 = tic; evalc('outLeg = mcmc().optimisation(y, mask, w, pars0, fitting, @obj.FWD, ''mcmc'', fitting);'); tLeg = toc(t0);
    rng(seedNew); parallel.gpu.rng(seedNew);
    evalc('outLegSame = mcmc().optimisation(y, mask, w, pars0, fitting, @obj.FWD, ''mcmc'', fitting);');

    fprintf('\n===== SNR %d =====\n', SNR);
    testCheck = isfield(outNew, 'diagnostics') && isfield(outNew.settings, 'forceNewPath') && outNew.settings.forceNewPath;
    fprintf('  new loop exercised (out.settings.forceNewPath): %d\n', testCheck);
    isPass = isPass && testCheck;

    % (a) KS
    rng(seedSubset + ksnr);
    sub  = sort(randperm(Nv, Nsub));
    pval = zeros(Nsub, numel(params));
    for kp = 1:numel(params)
        xA = outNew.posterior.(params{kp}); xB = outLeg.posterior.(params{kp});
        for kv = 1:Nsub
            [~, pval(kv,kp)] = ks2_asymptotic(thin_by_ess(xA(sub(kv),:,1)), thin_by_ess(xB(sub(kv),:,1)));
        end
    end
    nRej = nnz(pval < alpha);
    ok   = nRej >= lo && nRej <= hi;
    isPass = isPass && ok;
    fprintf('  (a) KS rejections %d/%d (%.3f) [M0 %d, R2star %d, noise %d], allowed [%d, %d] -> %s\n', ...
        nRej, nTest, nRej/nTest, nnz(pval(:,1)<alpha), nnz(pval(:,2)<alpha), nnz(pval(:,3)<alpha), lo, hi, passfail(ok));

    % (a) moments
    for kp = 1:numel(params)
        [zM, zS] = mc_compare(outNew.posterior.(params{kp}), outLeg.posterior.(params{kp}));
        fM = mean(abs(zM) > 1.96); fS = mean(abs(zS) > 1.96);
        ok = fM <= zFracMax && fS <= zFracMax && abs(median(zM)) <= zMedMax && abs(median(zS)) <= zMedMax;
        isPass = isPass && ok;
        fprintf('  (a) %-7s mean: frac|z|>1.96 %.3f, median z %+.3f | SD: frac %.3f, median z %+.3f -> %s\n', ...
            params{kp}, fM, median(zM), fS, median(zS), passfail(ok));
    end

    % (b) same-seed bitwise identity (information)
    same = true;
    for kp = 1:numel(params)
        same = same && isequal(outNew.posterior.(params{kp}), outLegSame.posterior.(params{kp}));
    end
    fprintf('  (b) same seed, posterior bitwise identical to legacy: %d (information)\n', same);

    % (c) timing (information)
    fprintf('  (c) wall time new %.1f s, legacy %.1f s, ratio %.2f (information)\n', tNew, tLeg, tNew/tLeg);
    fprintf('      median post-burn-in acceptance (new path) %.3f\n', median(outNew.diagnostics.acceptance(:)));
end

fprintf('\nT1 neutral vs legacy overall: %s   (total time %.1f s)\n', passfail(isPass), toc(tStart));

function s = passfail(c)
if c; s = 'PASS'; else; s = 'FAIL'; end
end
