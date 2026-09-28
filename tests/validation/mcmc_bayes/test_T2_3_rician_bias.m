%% test_T2_3_rician_bias.m
%
% T2.3 (Phase 2): characterisation only, NO pass/fail.
%
% The IVIM case of T2.1 (5 ground-truth voxels, ivim_t2_config.m, 16 b-values,
% 'marginal_S0noise', flat box prior on D, F, Dstar) with Rician noise
% (magnitude of complex Gaussian noise, sigma = S0/SNR per channel) at SNR 5
% and 10, alongside the Gaussian-noise equivalent (same seeds). The marginal
% likelihood assumes Gaussian noise, so the Rician rows show the model
% mismatch (noise floor) bias.
%
% Per truth voxel: 200 noise realisations (1000 voxels per run), one chain per
% voxel, parameterTransform 'sigmoid', joint updates, adaptStepSize,
% 40000 iterations, burn-in 10000, thinning 10.
%
% Reported per noise type, SNR, truth voxel and parameter (D, F, Dstar, S0, sigma):
%   bias of the posterior mean   = mean over realisations of (posterior mean   - truth)
%   bias of the posterior median = mean over realisations of (posterior median - truth)
%   (also relative to truth, %), the MC standard error of these biases
%   (SD over realisations / sqrt(200)), and the fraction of realisations whose
%   central 95% credible interval contains the truth (coverage, information only).
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T2_3_rician_bias
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%

clearvars; tStart = tic;

SNRs        = [5 10];
noiseTypes  = {'gaussian','rician'};
Nreal       = 200;
seedData    = [81 82];      % per SNR, shared by both noise types
seedStart   = 91;
seedRun     = 92;

cfg     = ivim_t2_config();
NvT     = numel(cfg.truth.D);
params  = {'D','F','Dstar','S0','noise'};

fprintf('T2.3 Rician bias (characterisation, no pass/fail). SNR %s, %d realisations per truth voxel\n', mat2str(SNRs), Nreal);
fprintf('Seeds: data %s (per SNR, same for both noise types), start %d, run %d\n\n', mat2str(seedData), seedStart, seedRun);

for ks = 1:numel(SNRs)
    SNR = SNRs(ks);
    for kn = 1:numel(noiseTypes)
        y = ivim_sim(cfg, SNR, Nreal, seedData(ks), noiseTypes{kn});     % [NvT*Nreal, Nb], truth index fastest
        Nv = size(y,1);

        f = struct();
        f.modelParams = cfg.modelParams; f.lb = cfg.lb; f.ub = cfg.ub; f.xStepSize = cfg.xStepSize;
        f.algorithm = 'MH';
        f.likelihood = 'marginal_S0noise'; f.S0Param = 'S0';
        f.parameterTransform = 'sigmoid';
        f.updateScheme = 'joint'; f.adaptStepSize = true; f.adaptInterval = 50;
        f.iteration = 40000; f.burnin = 10000; f.thinning = 10;
        f.metric = {'mean'};
        rng(seedStart);
        for k = 1:numel(f.modelParams)
            x0.(f.modelParams{k}) = f.lb(k) + (f.ub(k)-f.lb(k)) * (0.1 + 0.8*rand(Nv,1));
        end
        [post, out, tRun] = run_chains(y, f, @(p) ivim_fwd(p, cfg.b), x0, 1, seedRun);

        fprintf('--- %s noise, SNR %d: run %.1f s, median acceptance %.3f ---\n', noiseTypes{kn}, SNR, tRun, median(out.diagnostics.acceptance(:)));
        fprintf('%-6s %-5s %9s | %10s %7s %8s | %10s %7s %8s | %6s\n', 'param', 'voxel', 'truth', ...
            'bias(mean)', 'rel%', 'se', 'bias(med)', 'rel%', 'se', 'cov95');
        for kp = 1:numel(params)
            p = params{kp};
            xs = double(post.(p));                      % [Nv, Ns]
            pm = mean(xs, 2); pmed = median(xs, 2);
            xsort = sort(xs, 2); Ns = size(xs,2);
            lo = xsort(:, max(1,ceil(0.025*Ns))); hi = xsort(:, ceil(0.975*Ns));
            for v = 1:NvT
                idx = v:NvT:Nv;
                if strcmp(p,'noise'); tr = cfg.truth.S0(v)/SNR; else; tr = cfg.truth.(p)(v); end
                bM  = mean(pm(idx) - tr);   sM  = std(pm(idx))/sqrt(Nreal);
                bMd = mean(pmed(idx) - tr); sMd = std(pmed(idx))/sqrt(Nreal);
                cov = mean(lo(idx) <= tr & hi(idx) >= tr);
                fprintf('%-6s %-5d %9.4f | %10.4f %7.1f %8.4f | %10.4f %7.1f %8.4f | %6.3f\n', p, v, tr, ...
                    bM, 100*bM/tr, sM, bMd, 100*bMd/tr, sMd, cov);
            end
        end
        fprintf('\n');
    end
end
fprintf('T2.3 done (characterisation only, no pass/fail). Total time %.1f s\n', toc(tStart));
