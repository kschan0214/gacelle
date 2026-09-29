%% Example_r2star_mcmc_bayes.m
%
% Bayesian priors with the EXPERIMENTAL mcmc_bayes sampler on a simulated R2* phantom:
%   1. flat prior (voxel-wise, noise and M0 integrated out)
%   2. BSP: hierarchical (population) prior
%   3. BSP + spatial prior: two-stage empirical Bayes with a 3D MRF prior
% Used by the tutorial docs/tutorial/mcmc_bayes_tutorial.rst.
%
% Kwok-Shing Chan
% kchan2@mgh.harvard.edu
%
% Date created: 28 September 2026
%
addpath('../../gacelle'); addpath_gacelle; % this is the path to 'gacelle' package
clear;

%% Simulate a smooth two-tissue R2* phantom: 6 echoes at moderate SNR
seed = 23439; rng(seed); gpurng(seed);
dims    = [32 32 4];
te      = [3 9 15 21 27 33] * 1e-3;                                  % s
SNR     = 20;                                                   % at the first echo

[xx, yy] = ndgrid(linspace(-1,1,dims(1)), linspace(-1,1,dims(2)));
tissue  = repmat(xx.^2 + yy.^2 < 0.4, [1 1 dims(3)]);           % 'grey matter' disc in 'white matter'
smooth  = repmat(sin(pi*xx) .* cos(pi*yy), [1 1 dims(3)]);      % slow spatial variation

truth.R2star = 20 + 8*tissue + 4*smooth;                        % 1/s
truth.M0     = 1 + 0.1*smooth;

tt  = reshape(te, 1, 1, 1, []);
s   = truth.M0 .* exp(-tt .* truth.R2star);
y   = s + randn(size(s)) / SNR;
mask = true(dims);

%% Settings shared by all fits
fitting                     = [];
fitting.solver              = 'mcmc';
fitting.mcmcClass           = 'mcmc_bayes';          % experimental Bayesian sampler
fitting.algorithm           = 'MH';
fitting.iteration           = 1e4;
fitting.burnin              = 5e3;                   % adaptation during burn-in only
fitting.thinning            = 5;
fitting.repetition          = 4;                     % 4 chains for R-hat
fitting.overdisp            = 0.01;                  % over-dispersed chain starts
fitting.metric              = {'median','std'};
fitting.start               = 'default';
fitting.parameterTransform  = 'sigmoid';
fitting.adaptStepSize       = true;
fitting.likelihood          = 'marginal_S0noise_flat';   % noise and M0 integrated out
fitting.S0Param             = 'M0';

%% 1. Flat prior
out_flat = gpuR2starMapping(te).estimate(y, mask, fitting);

%% 2. BSP: hierarchical prior on R2*
fitting.prior               = [];
fitting.prior.hierarchical  = struct('params', {{'R2star'}});
out_bsp  = gpuR2starMapping(te).estimate(y, mask, fitting);

%% 3. BSP + spatial prior (two-stage empirical Bayes)
fitting.prior.mrf           = struct('mode', '3d', 'tau', 3);
out_mrf  = gpuR2starMapping(te).estimate(y, mask, fitting);

%% Compare with the truth
res = {out_flat, out_bsp, out_mrf}; names = {'flat','BSP','BSP + MRF'};
fprintf('%-10s %10s %10s %10s\n', '', 'RMSE', '90% cov.', 'R-hat');
for k = 1:numel(res)
    e    = res{k}.median.R2star(mask) - truth.R2star(mask);
    q    = prctile(reshape(res{k}.posterior.R2star, nnz(mask), []), [5 95], 2);   % 90% credible interval
    cov  = mean(truth.R2star(mask) >= q(:,1) & truth.R2star(mask) <= q(:,2));
    fprintf('%-10s %10.2f %10.2f %10.3f\n', names{k}, sqrt(mean(e.^2)), cov, median(res{k}.diagnostics.rhat.R2star(mask)));
end

%% Convergence of the population prior (transformed space)
disp(out_bsp.hyper.mean.mu);            % population mean of logit((R2* - lb)/(ub - lb))
disp(out_bsp.hyper.rhat.mu);            % aim for <= 1.01
disp(out_mrf.stage1.hyper.rhat.mu);     % two-stage: stage-1 population prior

%% Show the maps (middle slice)
figure;
lim = [min(truth.R2star(:)) max(truth.R2star(:))];
subplot(1,4,1); imagesc(truth.R2star(:,:,2), lim); axis image off; title('truth');
for k = 1:3
    subplot(1,4,1+k); imagesc(res{k}.median.R2star(:,:,2), lim); axis image off; title(names{k});
end
