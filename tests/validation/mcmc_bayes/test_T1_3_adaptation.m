%% test_T1_3_adaptation.m
%
% T1.3 (Phase 1): step-size adaptation.
%
% Monoexponential R2* model (gpuR2starMapping FWD), 1000 voxels, SNR 20
% and 50, 12 echoes, legacy target (Gaussian likelihood with sampled
% noise, flat box prior), parameterTransform = 'linear' so only the
% adaptation differs from legacy. Start points are the class defaults
% [M0 R2star noise] = [1 30 0.05]; the initial step is the class default
% xStepSize.
%
% Runs per SNR
%   J : updateScheme 'joint',         adaptStepSize, target 0.234
%   C : updateScheme 'componentwise', adaptStepSize, target 0.44
%   R : reference, legacy mcmc (non-adaptive, xStepSize), long run
%
% Criteria (stated before running), for each SNR
%   (a) acceptance: post-burn-in acceptance within target +/- 0.05 in
%       >= 95% of voxels; for C this must hold for EACH parameter.
%   (b) posterior unchanged: for J vs R and C vs R, per voxel and
%       parameter, z-scores of the posterior mean and SD difference
%       (mc_compare.m; MC standard errors from ESS, the SD's from the ESS
%       of (x-m)^2). Under equal posteriors z ~ N(0,1). PASS if, for each
%       parameter and each of mean/SD, the fraction |z| > 1.96 is <= 0.075
%       (nominal 0.05; the exact 99.5% binomial upper quantile for n=1000
%       is ~0.068, plus slack for ESS-estimation error) AND |median z| <=
%       0.15 (bias check; SE of the median of 1000 N(0,1) is ~0.04).
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T1_3_adaptation
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%

clearvars; tStart = tic;

% settings
Nv          = 1000;
SNRs        = [20 50];
seedData    = [20260926 20260927];  % per SNR (same data as T1.2)
seedRun     = [2001 2002 2003];     % J, C, R (rng and parallel.gpu.rng)

accTol      = 0.05;
accFrac     = 0.95;
zFracMax    = 0.075;
zMedMax     = 0.15;

% adaptive runs
itA         = 25000;  burnA = 5000;   thinA = 5;    adaptInterval = 50;
% reference (legacy, non-adaptive)
itR         = 200000; burnR = 20000;  thinR = 20;

fprintf('T1.3 adaptation: Nv=%d, SNR=%s\n', Nv, mat2str(SNRs));
fprintf('  adaptive: iteration=%d, burnin=%d, thinning=%d, adaptInterval=%d (%d windows)\n', itA, burnA, thinA, adaptInterval, floor(burnA/adaptInterval));
fprintf('  reference (legacy mcmc): iteration=%d, burnin=%d, thinning=%d\n', itR, burnR, thinR);
fprintf('Seeds: data %s, run (J, C, R) %s\n', mat2str(seedData), mat2str(seedRun));
fprintf('Criteria: (a) |acc - target| <= %.2f in >= %.0f%% voxels (per parameter for C);\n', accTol, 100*accFrac);
fprintf('          (b) frac(|z|>1.96) <= %.3f and |median z| <= %.2f for mean and SD, per parameter, J vs R and C vs R\n', zFracMax, zMedMax);

isPass = true;
for ksnr = 1:numel(SNRs)
    SNR = SNRs(ksnr);
    [y, mask, pars0, fitting, obj] = r2star_sim(Nv, SNR, seedData(ksnr));
    w = ones(size(y), 'single');
    params = fitting.modelParams;

    % J and C
    schemes = {'joint', 'componentwise'};
    targets = [0.234 0.44];
    out     = cell(1,3); tRun = zeros(1,3);
    for ks = 1:2
        f = fitting;
        f.iteration     = itA; f.burnin = burnA; f.thinning = thinA; f.repetition = 1;
        f.parameterTransform = 'linear';
        f.updateScheme  = schemes{ks};
        f.adaptStepSize = true;
        f.adaptInterval = adaptInterval;
        rng(seedRun(ks)); parallel.gpu.rng(seedRun(ks));
        t0 = tic;
        evalc('out{ks} = mcmc_bayes().optimisation(y, mask, w, pars0, f, @obj.FWD, ''mcmc'', f);');
        tRun(ks) = toc(t0);
    end
    % R
    f = fitting; f.iteration = itR; f.burnin = burnR; f.thinning = thinR; f.repetition = 1;
    rng(seedRun(3)); parallel.gpu.rng(seedRun(3));
    t0 = tic;
    evalc('out{3} = mcmc().optimisation(y, mask, w, pars0, f, @obj.FWD, ''mcmc'', f);');
    tRun(3) = toc(t0);

    fprintf('\n===== SNR %d (run time J %.1f s, C %.1f s, R %.1f s) =====\n', SNR, tRun);

    % (a) acceptance
    for ks = 1:2
        acc = reshape(out{ks}.diagnostics.acceptance, Nv, []);   % [Nv, Nblock]
        blocks = out{ks}.diagnostics.acceptanceBlocks;
        for kb = 1:size(acc,2)
            frac = mean(abs(acc(:,kb) - targets(ks)) <= accTol);
            ok   = frac >= accFrac;
            isPass = isPass && ok;
            fprintf('  (a) %-13s %-7s acceptance median %.3f [5%%,95%%] = [%.3f, %.3f], within %.3f+/-%.2f: %.1f%% -> %s\n', ...
                schemes{ks}, blocks{kb}, median(acc(:,kb)), quantile_(acc(:,kb),0.05), quantile_(acc(:,kb),0.95), ...
                targets(ks), accTol, 100*frac, passfail(ok));
        end
    end

    % (b) posterior agreement with the reference
    for ks = 1:2
        for kp = 1:numel(params)
            [zM, zS] = mc_compare(out{ks}.posterior.(params{kp}), out{3}.posterior.(params{kp}));
            fM = mean(abs(zM) > 1.96); fS = mean(abs(zS) > 1.96);
            ok = fM <= zFracMax && fS <= zFracMax && abs(median(zM)) <= zMedMax && abs(median(zS)) <= zMedMax;
            isPass = isPass && ok;
            fprintf('  (b) %-13s vs R %-7s mean: frac|z|>1.96 %.3f, median z %+.3f | SD: frac %.3f, median z %+.3f -> %s\n', ...
                schemes{ks}, params{kp}, fM, median(zM), fS, median(zS), passfail(ok));
        end
    end

    % information only: sampling efficiency (median ESS and ESS per second)
    names = {'J','C','R'};
    for ks = 1:3
        e = zeros(1,numel(params));
        for kp = 1:numel(params); e(kp) = median(mcmc_bayes.ess(out{ks}.posterior.(params{kp}))); end
        fprintf('  (info) %s median ESS [%s] = %s, ESS/s = %s\n', names{ks}, strjoin(params(:).',','), ...
            mat2str(round(e)), mat2str(round(e/tRun(ks),1)));
    end
end

fprintf('\nT1.3 overall: %s   (total time %.1f s)\n', passfail(isPass), toc(tStart));

function s = passfail(c)
if c; s = 'PASS'; else; s = 'FAIL'; end
end

function q = quantile_(x, p)
x = sort(x(:)); q = x(max(1, ceil(p*numel(x))));
end
