%% test_T4_5_mixing_vs_tau.m
%
% T4.5 (Phase 4): REPORT ONLY (no pass/fail). Mixing of the chromatic MRF sampler as a
% function of tau, L1 potential, on a small IVIM phantom (ivim_mrf_phantom.m, SNR 30,
% 'marginal_S0noise', u = [log D, logit F, log Dstar]).
%
% Prior: hierarchical fixed with ORACLE hyperparameters (mu, Sigma = mean and covariance of the
%   true u over the mask, + 1e-4 I), times the L1 MRF with W = 1./sqrt(diag(Sigma)) (default),
%   w_ij = 1. tau in {0.03, 0.1, 0.3, 1, 3, 10} and a baseline without MRF (hierarchical only).
% Modes: '3d' face (2 colours) and '2d' full r = 2 (9 colours).
% Sampler: joint updates, adaptStepSize (burn-in only), 10000 iterations, burn-in 2000,
%   thinning 4 (2000 draws per chain), 4 chains as stacked copies (one call; over-dispersed starts
%   u = mu + sd .* randn per chain).
% Reported per (mode, tau), on u (voxel-wise, all 3 parameters): split-R-hat (median, 95th pct,
%   fraction > 1.01), multi-chain ESS (median, 5th pct), ESS per 1000 post-burn-in iterations per
%   chain (median ESS / (4 x 8000) x 1000), ESS per second (median ESS / wall time of the call,
%   i.e. all voxels and 4 chains), median acceptance; plus (information) the RMSE of the posterior
%   mean of D, F, Dstar against the truth.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T4_5_mixing_vs_tau
% MCMC_BAYES_T4_ITER overrides the number of iterations (pilot; burn-in 20%, thinning 4).
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%

clearvars; tStart = tic;

iteration = 10000;
if ~isempty(getenv('MCMC_BAYES_T4_ITER')); iteration = str2double(getenv('MCMC_BAYES_T4_ITER')); end
burnin = round(0.2*iteration); thinning = 4; Nc = 4; SNR = 30;
taus  = [0.03 0.1 0.3 1 3 10 Inf];
modes = { {'3d', 1, 'face'}, {'2d', 2, 'full'} };
seedPhantom = 301; seedStart = 302; seedRun = 310;

[y, mask, truth, f] = ivim_mrf_phantom(SNR, seedPhantom);
b = f.b; f = rmfield(f, 'b');
f.iteration = iteration; f.burnin = burnin; f.thinning = thinning;
idx = find(mask); Nv = numel(idx); dims = size(mask); Nb = numel(b);
uT  = [log(truth.D(idx)) log(truth.F(idx)./(1-truth.F(idx))) log(truth.Dstar(idx))].';
mu  = mean(uT, 2); Sigma = cov(uT.') + 1e-4*eye(3); sd = sqrt(diag(Sigma));
fprintf('T4.5 mixing vs tau (report only): IVIM phantom %s, Nv %d, SNR %d, L1, oracle mu = %s, sd = %s\n', ...
    mat2str(dims), Nv, SNR, mat2str(mu.',3), mat2str(sd.',3));
fprintf('%d iterations, burn-in %d, thinning %d, %d chains (stacked); seeds phantom %d, starts %d, runs %d+k\n\n', ...
    iteration, burnin, thinning, Nc, seedPhantom, seedStart, seedRun);

% stacked copies with one empty slice between them, over-dispersed starts
Z = Nc*dims(3) + (Nc-1);
mask4 = false([dims(1:2) Z]); yy = zeros([dims(1:2) Z Nb]);
rng(seedStart);
x0 = struct('S0', ones([dims(1:2) Z]), 'D', ones([dims(1:2) Z]), 'F', 0.1*ones([dims(1:2) Z]), 'Dstar', 20*ones([dims(1:2) Z]), ...
            'noise', 0.03*ones([dims(1:2) Z]));
for c = 1:Nc
    zs = (c-1)*(dims(3)+1) + (1:dims(3));
    mask4(:,:,zs) = mask; yy(:,:,zs,:) = y;
    u0 = mu + sd .* randn(3, Nv);
    tmp = zeros(dims); tmp(idx) = exp(u0(1,:));            D = x0.D;     D(:,:,zs) = tmp;     x0.D = D;
    tmp = zeros(dims); tmp(idx) = 1./(1+exp(-u0(2,:)));    F = x0.F;     F(:,:,zs) = tmp;     x0.F = F;
    tmp = zeros(dims); tmp(idx) = exp(u0(3,:));            Ds = x0.Dstar; Ds(:,:,zs) = tmp;   x0.Dstar = Ds;
end
fwd = @(pp) ivim_fwd(pp, b);
names = {'D','F','Dstar'};

fprintf('%-4s %6s %8s %8s %8s %9s %8s %9s %9s %9s %6s | %7s %7s %7s\n', 'mode', 'tau', 'Rh med', 'Rh p95', 'Rh>1.01', 'ESS med', 'ESS p5', 'ESS/1k it', 'ESS/s', 'time s', 'acc', 'RMSE D', 'RMSE F', 'RMSE D*');
k = 0; rows = [];
for km = 1:numel(modes)
    [mode, r, conn] = modes{km}{:};
    for tau = taus
        k = k + 1;
        g = f;
        g.prior.hierarchical = struct('fixed', true, 'mu', mu, 'Sigma', Sigma);
        if isfinite(tau)
            g.prior.mrf = struct('potential', 'l1', 'tau', tau, 'mode', mode, 'radius', r, 'connectivity', conn);
        elseif km > 1
            continue                                       % baseline without MRF once
        end
        rng(seedRun + k); parallel.gpu.rng(seedRun + k);
        t0 = tic;
        evalc('out = mcmc_bayes().optimisation(yy, mask4, [], x0, g, fwd);');
        tRun = toc(t0);
        Ns  = size(out.posterior.D, 2);
        uS  = zeros(Nv, Ns, Nc, 3);
        xs  = {out.posterior.D, out.posterior.F, out.posterior.Dstar};
        tf  = {@log, @(x) log(x./(1-x)), @log};
        Rh = zeros(Nv, 3); Es = zeros(Nv, 3); rmse = zeros(1, 3);
        tv = {truth.D(idx), truth.F(idx), truth.Dstar(idx)};
        for p = 1:3
            x = permute(reshape(double(xs{p}), Nv, Nc, Ns), [1 3 2]);
            uS(:,:,:,p) = tf{p}(x);
            Rh(:,p) = mcmc_bayes.rhat(uS(:,:,:,p));
            Es(:,p) = mcmc_bayes.ess(uS(:,:,:,p));
            rmse(p) = sqrt(mean((mean(reshape(x, Nv, []), 2) - tv{p}).^2));
        end
        acc = reshape(out.diagnostics.acceptance, numel(mask4), []); acc = median(acc(mask4(:)));
        essMed = median(Es(:)); Nit = (iteration - burnin);
        row = [km tau median(Rh(:)) prctile_(Rh(:), 95) mean(Rh(:) > 1.01) essMed prctile_(Es(:), 5) ...
               essMed/(Nc*Nit)*1000 essMed/tRun tRun acc rmse];
        rows = [rows; row]; %#ok<AGROW>
        if isfinite(tau); md = mode; else; md = 'none'; end
        fprintf('%-4s %6g %8.4f %8.4f %8.3f %9.0f %8.0f %9.2f %9.2f %9.0f %6.3f | %7.4f %7.4f %7.3f\n', md, tau, row(3:end));
    end
end
save(fullfile(tempdir, 'T4_5_mixing_vs_tau.mat'), 'rows', 'taus', 'modes');
fprintf('\nT4.5: reported (no pass/fail criterion). Total time %.1f min\n', toc(tStart)/60);

function q = prctile_(x, p)
% percentile without the Statistics toolbox (nearest rank)
x = sort(x(:)); q = x(max(1, min(numel(x), ceil(p/100*numel(x)))));
end
