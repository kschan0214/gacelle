%% demo_convergence_usage.m
%
% Demonstrates askadam.m's v1.1 convergence options (see
% docs/advanced/askadam_convergence.rst) by fitting the same tiny
% synthetic NEXI dataset under 6 different convergence configurations -
% the default linear-slope convergence model, EMA smoothing, outlier-
% robust convergence (on its own and combined with EMA), step-norm
% convergence, and all of the above combined - and comparing:
%   1. how many iterations each config took and its final loss (printed
%      summary table);
%   2. how well each recovered the ground-truth parameters (scatter of
%      fitted vs. GT, one row per parameter, one column per config,
%      including the unfitted starting point for reference);
%   3. each config's per-sample residual loss;
%   4. each config's per-parameter RMSE against ground truth (bar chart;
%      the unfitted starting point is deliberately excluded here since
%      its RMSE would dwarf every fitted config and make the comparison
%      between them unreadable).
%
% All 6 configs are fitted to IDENTICAL noisy data (same rng seed reset
% before every call) and start from the same 'likelihood'-based initial
% guess, so any difference between them is purely due to the convergence
% option(s) each one changes relative to fitting_base.
%
% Kwok-Shing Chan
% kchan2@mgh.harvard.edu
%
% Date created: 12 June 2026
% Date modified:
%
addpath('../../gacelle'); addpath_gacelle; % this is the path to 'gacelle' package
clear;

%% ========================================================
% Simulate data
% ========================================================
% A single-protocol (15-measurement, 3-shell) synthetic NEXI dataset,
% following the same recipe as NEXI/demo_gpuNEXI_NoisePropagation.m -
% see that script for the full ground-truth-generation/noise details.
seed    = 23439; rng(seed); gpurng(seed);
Nsample = 1e3;
SNR     = 20;

bval_sorted   = [2.3, 3.5, 4.8, 6.5, 2.3, 3.5, 4.8, 6.5, 11.0, 2.3, 3.5, 4.8, 6.5, 11.0, 17.5];
BDELTA_sorted = [13, 13, 13, 13, 21, 21, 21, 21, 21, 30, 30, 30, 30, 30, 30];

tex_range = [2, 50];  fa_range = [0.1, 0.8];
Da_range  = [1.5, 3]; De_range = [0.5, 1.5];
p2_range  = [0.05, 0.5];

tex_GT = single(rand(1,Nsample) * diff(tex_range) + min(tex_range));
Da_GT  = single(rand(1,Nsample) * diff(Da_range)  + min(Da_range));
fa_GT  = single(rand(1,Nsample) * diff(fa_range)  + min(fa_range));
De_GT  = single(rand(1,Nsample) * diff(De_range)  + min(De_range));
p2_GT  = single(rand(1,Nsample) * diff(p2_range)  + min(p2_range));
ra_GT  = (1-fa_GT) ./ tex_GT;

pars    = [];
pars.fa = fa_GT; pars.Da = Da_GT; pars.De = De_GT;
pars.ra = ra_GT; pars.p2 = p2_GT;

lmax   = 2;
objGPU = gpuNEXI(bval_sorted, BDELTA_sorted);
s      = objGPU.FWD(pars, lmax);

noiseLv = 1/SNR;
y       = s + randn(size(s)) .* noiseLv;
y       = permute(y, [3 2 4 1]);
mask    = ones(size(y,1:3)) > 0;

%% ========================================================
% Base fitting settings (shared across all configurations)
% ========================================================
% Every config below starts from a COPY of fitting_base and only adds/
% overrides the convergence-specific field(s) it's testing - everything
% else (solver, learning rate, loss function, starting-point method, ...)
% stays fixed so the comparison isolates the effect of convergence
% settings alone.
fitting_base                    = objGPU.check_set_default([]);
fitting_base.solver             = 'askadam';
fitting_base.iteration          = 4000;
fitting_base.initialLearnRate   = 0.001;
fitting_base.convergenceValue   = 1e-8;
fitting_base.lossFunction       = 'l1';
fitting_base.tol                = 1e-8;
fitting_base.isDisplay          = false;
fitting_base.lmax               = lmax;
fitting_base.start              = 'likelihood';

%% ========================================================
% Config 1: Original behaviour (mode 1, linear convergence)
% ========================================================
fprintf('\n=== Config 1: Original (linear convergence) ===\n');
rng(seed); gpurng(seed);
fitting             = fitting_base;
fitting.convergenceModel = 'linear';   % default, explicit for clarity
fitting.patience    = 5;
out_linear          = objGPU.estimate(y, mask, [], fitting);

%% ========================================================
% Config 2: EMA convergence model only
% Tests whether EMA smoothing changes convergence behaviour
% relative to the linear slope baseline
% ========================================================
fprintf('\n=== Config 2: EMA convergence model ===\n');
rng(seed); gpurng(seed);
fitting                     = fitting_base;
fitting.convergenceModel    = 'ema';
fitting.emaDecay            = 0.95;
fitting.patience            = 5;
out_ema                     = objGPU.estimate(y, mask, [], fitting);

%% ========================================================
% Config 3: Robust convergence only (with linear model)
% Tests whether outlier-aware convergence and gradient
% downweighting changes results vs. baseline
% ========================================================
% fitting.robustConvergence and the fitting.outlier* fields below
% together control per-voxel outlier detection/downweighting during
% convergence checking (see docs/advanced/askadam_convergence.rst for
% the full description of each):
%   .outlierWeight          - downweight applied to flagged voxels
%   .weightUpdateInterval   - iterations between outlier-weight updates
%   .outlierCheckWindow     - iterations averaged over when checking
%   .outlierMinFlagDuration - min iterations a voxel must stay flagged
%   .outlier{Voxel,Pop}Thres      - steady-state flagging thresholds
%   .outlierInit{Thres,PopThres}  - (usually looser) early-iteration
%                                    thresholds, before the fit settles
fprintf('\n=== Config 3: Robust convergence (linear) ===\n');
rng(seed); gpurng(seed);
fitting                         = fitting_base;
fitting.convergenceModel        = 'linear';
fitting.robustConvergence       = true;
fitting.outlierWeight           = 0.1;
fitting.weightUpdateInterval    = 5;
fitting.outlierCheckWindow      = 5;
fitting.outlierMinFlagDuration  = 5;
fitting.outlierVoxelThres       = 0.01;
fitting.outlierPopThres         = 0.05;
fitting.outlierInitThres        = 0.05;
fitting.outlierInitPopThres     = 0.20;
out_robust                      = objGPU.estimate(y, mask, [], fitting);

%% ========================================================
% Config 4: Robust convergence + EMA (full mode 2)
% Tests the combined effect of both new features
% (fitting.outlier* fields as described under Config 3 above)
% ========================================================
fprintf('\n=== Config 4: Robust convergence + EMA ===\n');
rng(seed); gpurng(seed);
fitting                         = fitting_base;
fitting.convergenceModel        = 'ema';
fitting.emaDecay                = 0.95;
fitting.robustConvergence       = true;
fitting.outlierWeight           = 0.1;
fitting.weightUpdateInterval    = 5;
fitting.outlierCheckWindow      = 5;
fitting.outlierMinFlagDuration  = 5;
fitting.outlierVoxelThres       = 0.01;
fitting.outlierPopThres         = 0.05;
fitting.outlierInitThres        = 0.05;
fitting.outlierInitPopThres     = 0.20;
out_robust_ema                  = objGPU.estimate(y, mask, [], fitting);

%% ========================================================
% Config 5: Step norm convergence signal
% Tests whether step norm catches stagnation not caught
% by loss-based signal alone
% ========================================================
% fitting.convergenceStepTol: converged once the relative change in
% parameter values between iterations drops below this for
% fitting.patienceStep consecutive checks - independent of the loss-based
% criteria (fitting.convergenceModel/robustConvergence above), and can
% trigger stopping even if the loss-based signal hasn't.
fprintf('\n=== Config 5: Step norm convergence signal ===\n');
rng(seed); gpurng(seed);
fitting                         = fitting_base;
fitting.convergenceModel        = 'linear';
fitting.convergenceStepTol      = 1e-6;
fitting.patienceStep            = 5;
out_step                        = objGPU.estimate(y, mask, [], fitting);

%% ========================================================
% Config 6: All signals combined
% Tests the full v1.1 feature set together
% (EMA + robust convergence [Config 3/4] + step norm [Config 5])
% ========================================================
fprintf('\n=== Config 6: All signals combined ===\n');
rng(seed); gpurng(seed);
fitting                         = fitting_base;
fitting.convergenceModel        = 'ema';
fitting.emaDecay                = 0.95;
fitting.robustConvergence       = true;
fitting.outlierWeight           = 0.1;
fitting.weightUpdateInterval    = 5;
fitting.outlierCheckWindow      = 5;
fitting.outlierMinFlagDuration  = 5;
fitting.outlierVoxelThres       = 0.01;
fitting.outlierPopThres         = 0.05;
fitting.outlierInitThres        = 0.05;
fitting.outlierInitPopThres     = 0.20;
fitting.convergenceStepTol      = 1e-6;
fitting.patienceStep            = 5;
out_all                         = objGPU.estimate(y, mask, [], fitting);

%% ========================================================
% Reference: starting point (0 iterations)
% ========================================================
% fitting.iteration = 0 makes estimate() return the 'likelihood'
% starting-point estimate itself (no askadam updates applied) - used
% below as the "Start" column/baseline in the scatter and per-sample-loss
% plots, to show how much each config's fit actually improved on it.
rng(seed); gpurng(seed);
fitting_ref             = fitting_base;
fitting_ref.iteration   = 0;
pars0                   = objGPU.estimate(y, mask, [], fitting_ref);

%% ========================================================
% Summary: compare iteration counts and final loss
% ========================================================
fprintf('\n=== Summary ===\n');
fprintf('%-40s | Iterations | Final loss\n', 'Config');
fprintf('%s\n', repmat('-',1,65));
configs = {'1: linear (baseline)', '2: EMA', '3: robust+linear', ...
           '4: robust+EMA', '5: step norm', '6: all combined'};
outs    = {out_linear, out_ema, out_robust, out_robust_ema, out_step, out_all};
for k = 1:numel(outs)
    fprintf('%-40s | %10d | %.6e\n', configs{k}, outs{k}.final.Niteration, outs{k}.final.loss);
end

%% ========================================================
% Plot: scatter plots of fitted vs GT for each config
% Grid of fitted (y) vs. ground-truth (x) scatter plots: one row per
% model parameter (fa, Da, De, ra, p2, plus the derived exchange time
% 'tex'), one column per config (including the unfitted "Start" point
% for reference). Points on the black identity line (refline(1)) are
% recovered exactly.
% ========================================================
field = fieldnames(pars);
Nfield = numel(field) + 1;  % +1 for tex

config_labels = {'Start', 'linear', 'EMA', 'robust+linear', 'robust+EMA', 'step', 'all'};
all_outs      = [{pars0}, outs];

figure('Name','Fitted vs GT — all configs');
tiledlayout(Nfield, numel(all_outs), 'TileSpacing', 'compact');

for f = 1:numel(field)
    for k = 1:numel(all_outs)
        nexttile;
        scatter(pars.(field{f}), all_outs{k}.final.(field{f}), 3, 'filled', 'MarkerFaceAlpha', 0.3);
        hold on; h = refline(1); h.Color = 'k';
        if f == 1; title(config_labels{k}); end
        if k == 1; ylabel(field{f}); end
        axis tight;
    end
end

% tex row
for k = 1:numel(all_outs)
    nexttile;
    tex_fitted = (1 - all_outs{k}.final.fa) ./ all_outs{k}.final.ra;
    scatter((1-pars.fa)./pars.ra, tex_fitted, 3, 'filled', 'MarkerFaceAlpha', 0.3);
    hold on; h = refline(1); h.Color = 'k';
    if k == 1; ylabel('tex'); end
    axis tight;
end

%% ========================================================
% Plot: per-sample loss comparison across configs
% out.final.resloss is each config's final per-sample loss - overlaid
% here (one marker style/colour per config, "Start" included) to see
% whether any config leaves a systematically worse-fitting subpopulation
% of samples than the others.
% ========================================================
figure('Name', 'Per-sample residual loss — all configs');
tiledlayout(1, 1);
nexttile;
hold on;
colors = lines(numel(outs)+1);
plot(pars0.final.resloss, 'x', 'Color', colors(1,:), 'DisplayName', 'Start');
markers = {'o','+','s','d','^','v'};
for k = 1:numel(outs)
    plot(outs{k}.final.resloss, markers{k}, 'Color', colors(k+1,:), ...
        'DisplayName', config_labels{k+1});
end
legend; xlabel('Sample'); ylabel('Loss'); title('Per-sample residual loss');

%% ========================================================
% Plot: RMSE per parameter across configs
% One bar chart per parameter (including derived 'tex'), one bar per
% fitted config - the unfitted "Start" baseline is deliberately excluded
% here (its RMSE would dwarf every fitted config's and make the
% comparison between them unreadable); see the scatter plots above for
% Start-vs-fitted comparisons instead.
% ========================================================
figure('Name', 'RMSE per parameter — all configs');
tiledlayout(1, Nfield, 'TileSpacing', 'compact');

GT_vals = struct();
for f = 1:numel(field); GT_vals.(field{f}) = pars.(field{f}); end
GT_vals.tex = (1-pars.fa) ./ pars.ra;

all_field = [field; {'tex'}];
for f = 1:numel(all_field)
    nexttile; hold on;
    rmse_vals = zeros(1, numel(outs));
    for k = 1:numel(outs)
        if strcmp(all_field{f}, 'tex')
            fitted = (1 - outs{k}.final.fa) ./ outs{k}.final.ra;
        else
            fitted = outs{k}.final.(all_field{f});
        end
        rmse_vals(k) = sqrt(mean((fitted(:) - GT_vals.(all_field{f})(:)).^2));
    end
    bar(rmse_vals);
    set(gca, 'XTick', 1:numel(rmse_vals), 'XTickLabel', config_labels(2:end), 'XTickLabelRotation', 30);
    ylabel('RMSE'); title(all_field{f});
end