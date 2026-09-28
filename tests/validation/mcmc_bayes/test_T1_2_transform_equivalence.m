%% test_T1_2_transform_equivalence.m
%
% T1.2 (Phase 1): transform equivalence under a flat native prior.
%
% Monoexponential R2* model (gpuR2starMapping FWD), 1000 voxels, SNR 20
% and 50, 12 echoes. The target is the legacy one: Gaussian likelihood with
% sampled noise, uniform box prior on [lb,ub] in native space. Two arms
% sample the same target:
%   A: parameterTransform = 'linear'   (box by rejection)
%   B: parameterTransform = 'sigmoid'  (box by construction, + log-Jacobian)
% Both use joint updates with step-size adaptation during burn-in, and
% different seeds so the two chains are independent.
%
% Test: on a random subset of 200 voxels, for each of the 3 parameters, a
% two-sample KS test (asymptotic p-value, ks2_asymptotic.m) between the
% two arms' native-space samples. Each chain is thinned by 2*tau
% (tau = Ns/ESS, thin_by_ess.m) so retained samples are roughly
% independent.
%
% Criterion (stated before running): under H0 each test rejects at
% alpha = 0.05 with probability 0.05. With 200 voxels x 3 parameters =
% 600 tests per SNR, PASS if the number of rejections lies in the exact
% central 99% interval of Binomial(600, 0.05) (binom_bounds.m, [17, 45]
% computed at run time) for BOTH SNRs. Caveat: the 3 tests within a voxel are not
% independent, which widens the true spread of the count slightly.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T1_2_transform_equivalence
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
seedData    = [20260926 20260927];  % per SNR
seedRun     = [1001 1002];          % arm A, arm B (rng and parallel.gpu.rng)
seedSubset  = 7;

iteration   = 30000;
burnin      = 5000;
thinning    = 5;

fprintf('T1.2 transform equivalence: Nv=%d, subset=%d, SNR=%s, iteration=%d, burnin=%d, thinning=%d\n', ...
    Nv, Nsub, mat2str(SNRs), iteration, burnin, thinning);
fprintf('Seeds: data %s, run (linear, sigmoid) %s, subset %d\n', mat2str(seedData), mat2str(seedRun), seedSubset);

nTest       = Nsub * 3;
[lo, hi]    = binom_bounds(nTest, alpha, 0.01);
fprintf('Criterion: #rejections (alpha=%.2f) out of %d in [%d, %d] (central 99%% of Binomial(%d,%.2f)) for each SNR\n', ...
    alpha, nTest, lo, hi, nTest, alpha);

isPass = true(1, numel(SNRs));
for ksnr = 1:numel(SNRs)
    SNR = SNRs(ksnr);
    [y, mask, pars0, fitting, obj] = r2star_sim(Nv, SNR, seedData(ksnr));
    w = ones(size(y), 'single');

    fitting.iteration       = iteration;
    fitting.burnin          = burnin;
    fitting.thinning        = thinning;
    fitting.repetition      = 1;
    fitting.updateScheme    = 'joint';
    fitting.adaptStepSize   = true;
    fitting.adaptInterval   = 50;

    arms = {'linear', 'sigmoid'};
    out  = cell(1,2); tArm = zeros(1,2);
    for ka = 1:2
        f = fitting; f.parameterTransform = arms{ka};
        rng(seedRun(ka)); parallel.gpu.rng(seedRun(ka));
        t0 = tic;
        evalc('out{ka} = mcmc_bayes().optimisation(y, mask, w, pars0, f, @obj.FWD, ''mcmc'', f);');
        tArm(ka) = toc(t0);
    end

    rng(seedSubset + ksnr);
    sub  = sort(randperm(Nv, Nsub));

    params = fitting.modelParams;
    pval   = zeros(Nsub, numel(params));
    nKeep  = zeros(Nsub, numel(params), 2);
    for kp = 1:numel(params)
        xA = out{1}.posterior.(params{kp});
        xB = out{2}.posterior.(params{kp});
        for kv = 1:Nsub
            a = thin_by_ess(xA(sub(kv),:,1));
            b = thin_by_ess(xB(sub(kv),:,1));
            nKeep(kv,kp,:) = [numel(a) numel(b)];
            [~, pval(kv,kp)] = ks2_asymptotic(a, b);
        end
    end
    nRej        = nnz(pval < alpha);
    isPass(ksnr)= nRej >= lo && nRej <= hi;

    fprintf('\nSNR %d: run time linear %.1f s, sigmoid %.1f s\n', SNR, tArm(1), tArm(2));
    for kp = 1:numel(params)
        accA = out{1}.diagnostics.acceptance; accB = out{2}.diagnostics.acceptance;
        fprintf('  %-7s rejections %3d/%d (%.3f), median retained n (lin/sig) %d/%d\n', params{kp}, ...
            nnz(pval(:,kp) < alpha), Nsub, mean(pval(:,kp) < alpha), ...
            round(median(nKeep(:,kp,1))), round(median(nKeep(:,kp,2))));
    end
    fprintf('  median post-burn-in acceptance (lin/sig) %.3f/%.3f\n', median(accA(:)), median(accB(:)));
    fprintf('  total rejections %d/%d (%.3f), allowed [%d, %d] -> %s\n', nRej, nTest, nRej/nTest, lo, hi, ...
        string(ifelse(isPass(ksnr), 'PASS', 'FAIL')));
end

fprintf('\nT1.2 overall: %s   (total time %.1f s)\n', string(ifelse(all(isPass), 'PASS', 'FAIL')), toc(tStart));

function s = ifelse(c, a, b)
if c; s = a; else; s = b; end
end
