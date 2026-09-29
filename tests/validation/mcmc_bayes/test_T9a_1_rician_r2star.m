%% test_T9a_1_rician_r2star.m
%
% T9a.1 (Phase 9a): characterisation, NO pass/fail.
%
% Monoexponential R2* phantom (gpuR2starMapping, 12 echoes, te = 2-50 ms) with Rician noise
% (magnitude of complex Gaussian noise, sigma = M0_ref/SNR per channel, M0_ref = 1) at SNR 5 and 10.
% The late echoes of the high-R2* voxels sit in the noise floor, where a Gaussian likelihood
% overestimates the signal and so underestimates R2*. Compared likelihoods (same data, same seeds):
%   'rician'                  exact Rician density, sigma sampled
%   'rician (known sigma)'    exact Rician density, sigma fixed at the true value (fitting.ricianSigma)
%   'gaussian'                Gaussian, sigma sampled
%   'marginal_S0noise_flat'   Gaussian, M0 and sigma integrated out (flat M0, 1/sigma^2)
%
% Truth: M0 = 1, R2* in {20, 40, 70, 100, 150} 1/s, Nreal noise realisations per truth value.
% One chain per voxel, parameterTransform sigmoid (M0, R2*) and log (noise), joint updates,
% adaptStepSize, flat box prior (M0 in [0, 2], R2* in [0.1, 250], noise in [1e-3, 1]).
%
% Reported per SNR, likelihood and truth R2* (and for noise = sigma):
%   bias of the posterior mean (mean over realisations of mean - truth), its MC standard error,
%   bias of the posterior median, RMSE of the posterior mean, and the 90% coverage (fraction of central 90% credible intervals,
%   5% and 95% sample quantiles, that contain the truth; nominal 0.90, MC SE ~ sqrt(0.09/Nreal)).
%
% Sizes: full Nreal = 400, iteration 20000, burn-in 10000, thinning 10. The environment variable
% MCMC_BAYES_T9A_PILOT = 1 runs a pilot (Nreal = 20, iteration 2000, burn-in 1000, thinning 5)
% that only checks the script and gives the time per run.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T9a_1_rician_r2star
%
% Kwok-Shing Chan @ MGH
% Date created: 29 September 2026
% Date modified: 29 September 2026 (arm 'rician' with known sigma, fitting.ricianSigma)
%

clearvars; tStart = tic;

isPilot     = strcmp(getenv('MCMC_BAYES_T9A_PILOT'), '1');
SNRs        = [5 10];
R2true      = [20 40 70 100 150];
liks        = {'rician', 'rician (known sigma)', 'gaussian', 'marginal_S0noise_flat'};
seedData    = [301 302];        % per SNR, shared by all likelihoods
seedRun     = 303;
if isPilot
    Nreal = 20;  iteration = 2000;  burnin = 1000;  thinning = 5;
else
    Nreal = 400; iteration = 20000; burnin = 10000; thinning = 10;
end
te      = linspace(2e-3, 50e-3, 12);
obj     = gpuR2starMapping(te);
NvT     = numel(R2true);
Nv      = NvT*Nreal;

fprintf('T9a.1 Rician vs Gaussian likelihoods, R2* phantom (characterisation, no pass/fail)%s\n', repmat(' [PILOT]', 1, isPilot));
fprintf('12 echoes %.0f-%.0f ms, truth R2* %s 1/s, M0 = 1, SNR %s, %d realisations per truth value\n', ...
    1e3*te(1), 1e3*te(end), mat2str(R2true), mat2str(SNRs), Nreal);
fprintf('iteration %d, burn-in %d, thinning %d; seeds: data %s (per SNR), run %d\n\n', iteration, burnin, thinning, mat2str(seedData), seedRun);

for ks = 1:numel(SNRs)
    SNR   = SNRs(ks);
    sigma = 1/SNR;
    % data: truth index fastest
    truth.M0     = ones(1, Nv);
    truth.R2star = repmat(R2true, 1, Nreal);
    s     = double(gather(obj.FWD(truth))).';          % [Nv, Nte]
    rng(seedData(ks));
    y     = abs(s + sigma*randn(size(s)) + 1i*sigma*randn(size(s)));
    fprintf('=== SNR %d (sigma = %.3f): mean data - signal at the last echo, R2* = %d: %+.4f (noise floor) ===\n', ...
        SNR, sigma, R2true(end), mean(y(NvT:NvT:end, end) - s(NvT:NvT:end, end)));

    for kl = 1:numel(liks)
        f = struct();
        f.modelParams = {'M0';'R2star';'noise'};
        f.lb = [0; 0.1; 1e-3]; f.ub = [2; 250; 1]; f.xStepSize = [0.02; 2; 0.01];
        f.algorithm = 'MH'; f.likelihood = strtok(liks{kl});
        if strcmp(liks{kl}, 'marginal_S0noise_flat'); f.S0Param = 'M0'; end
        if strcmp(liks{kl}, 'rician (known sigma)'); f.ricianSigma = sigma; end    % noise fixed, not sampled
        f.parameterTransform = {'sigmoid','sigmoid','log'};
        f.updateScheme = 'joint'; f.adaptStepSize = true; f.adaptInterval = 50;
        f.iteration = iteration; f.burnin = burnin; f.thinning = thinning;
        f.metric = {'mean'};
        x0 = struct('M0', ones(Nv,1), 'R2star', 50*ones(Nv,1), 'noise', 0.1*ones(Nv,1));
        [post, out, tRun] = run_chains(y, f, @(p) obj.FWD(p), x0, 1, seedRun);

        fprintf('--- %s: run %.1f s (%.2f ms/iteration), median acceptance %.3f ---\n', liks{kl}, tRun, 1e3*tRun/iteration, ...
            median(out.diagnostics.acceptance(:)));
        fprintf('%-7s %7s | %9s %7s %8s | %9s | %8s | %6s\n', 'param', 'truth', 'bias', 'rel%', 'se', 'bias(med)', 'RMSE', 'cov90');
        params = {'R2star', 'M0', 'noise'};
        if isfield(f, 'ricianSigma'); params = params(1:2); end     % noise fixed at the truth: not reported
        for p = params
            xs  = double(post.(p{1}));                    % [Nv, Ns]
            pm  = mean(xs, 2);
            xsr = sort(xs, 2); Ns = size(xs, 2);
            pmd = xsr(:, max(1, round(0.5*Ns)));
            lo  = xsr(:, max(1, round(0.05*Ns))); hi = xsr(:, min(Ns, round(0.95*Ns)));
            for v = 1:NvT
                idx = v:NvT:Nv;
                switch p{1}
                    case 'R2star'; tr = R2true(v);
                    case 'M0';     tr = 1;
                    case 'noise';  tr = sigma;
                end
                b   = mean(pm(idx) - tr); se = std(pm(idx))/sqrt(Nreal); bmd = mean(pmd(idx) - tr);
                rm  = sqrt(mean((pm(idx) - tr).^2));
                cv  = mean(lo(idx) <= tr & hi(idx) >= tr);
                fprintf('%-7s %7.3g | %+9.4f %+7.1f %8.4f | %+9.4f | %8.4f | %6.3f   (R2* = %d)\n', p{1}, tr, b, 100*b/tr, se, bmd, rm, cv, R2true(v));
            end
        end
        fprintf('\n');
    end
end
fprintf('T9a.1 done (characterisation only, no pass/fail). Total time %.1f s\n', toc(tStart));
