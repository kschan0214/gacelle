%% demo_convergence_usage2.m
%
% A more realistic companion to demo_convergence_usage.m: demonstrates
% askadam.m's v1.1 convergence options (see
% docs/advanced/askadam_convergence.rst) on a full DWI acquisition
% protocol (32 gradient directions per shell across 3 b-value/diffusion-
% time shells, plus interspersed b0s) rather than a pre-averaged
% rotationally-invariant signal, and adds a deliberate 1% population of
% pure-noise "background" voxels (e.g. representing CSF/outside-brain)
% to test a robustness-relevant edge case: voxels with no real signal to
% fit at all.
%
% Same 6 convergence configurations as demo_convergence_usage.m (linear,
% EMA, robust+linear, robust+EMA, step norm, all combined - see that
% script or docs/advanced/askadam_convergence.rst for what each
% fitting.* option does), but adds:
%   - a background-vs-signal loss breakdown (summary table, section 6,
%     and the per-sample-loss plot's dashed boundary marker), to check
%     whether robust convergence helps by genuinely converging faster on
%     the signal voxels, rather than just by downweighting the noise
%     ones;
%   - RMSE/scatter comparisons computed on SIGNAL voxels only (the
%     background voxels have no ground truth to compare against).
%
% Requires the external `protocoldesign` toolbox (gradient-direction
% table generation - not part of GACELLE) already on the path.
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
% 1. Simulate data
% ========================================================
% Ground truth (fa, Da, De, ra, kappa) -> per-voxel Watson-dispersed NEXI
% signal on a real multi-shell/multi-direction protocol -> add b0s,
% Gaussian noise, and pure-noise background voxels -> reshape into
% GACELLE's expected [x,y,z,measurement] layout.
seed = 23439; rng(seed); gpurng(seed);

SNR     = 50;
Ngdir   = 32;
Nsample = 1e3;

% 3 diffusion-time shells (13/21/30 ms), each sampling an increasing
% number of the 6 unique b-values (4/5/6 respectively) - a realistic
% protocol shape, not a copy-paste of the same shell 3 times.
bval_unique  = [2.3 3.5 4.8 6.5 11.5 17.5];
delta_little = 6;
DELTA_big    = [13, 21, 30];
Nshell       = [4, 5, 6];
bval         = [bval_unique(1:Nshell(1)) ...
                bval_unique(1:Nshell(2)) ...
                bval_unique(1:Nshell(3))].';

% gradient direction table (external `protocoldesign` toolbox - see the
% header requirement note above)
method = 'matlab';
pd     = protocoldesign();
bvec   = pd.dirgen(Ngdir, method);

Delta = [DELTA_big(1)*ones(Nshell(1),1); ...
         DELTA_big(2)*ones(Nshell(2),1); ...
         DELTA_big(3)*ones(Nshell(3),1)];
delta = delta_little * ones(numel(bval), 1);

% ground truth parameters: fa, Da, De, ra (via tex), kappa (Watson
% orientation dispersion concentration - higher kappa = more aligned/less
% dispersed fibres), one column per voxel
intervals = [0.1  0.9  ; ...   % fa
             1.5  3.0  ; ...   % Da
             0.5  1.5  ; ...   % De
             2.0  50.0 ; ...   % tex -> converted to ra below
             0.5  1.5 ];      % kappa

pars      = intervals(:,1) + rand(size(intervals,1), Nsample) .* diff(intervals, [], 2);
pars(4,:) = max((1./pars(4,:)) .* (1-pars(1,:)), 1/200);  % tex -> ra

% enforce Da >= De: NEXI's intra-/extra-neurite diffusivities are only
% identifiable up to a label swap (fa<->1-fa, Da<->De describe the same
% signal), so ground truth is generated with independent random Da/De
% then canonically sorted here to avoid comparing a fit against the
% "wrong" (swapped) ground-truth label.
tmp      = pars;
ind      = find(pars(2,:) < pars(3,:));
pars(2,ind) = tmp(3,ind);
pars(3,ind) = tmp(2,ind);

% forward signal generation: per-voxel Watson-dispersed NEXI signal via
% spherical-harmonic convolution (NEXIrotinv.WatsonSH + .NEXIsh +
% .SHconv), up to lmax=8 - much higher angular resolution than the
% lmax=2 used for fitting below, so the simulated signal isn't
% artificially limited to what the fit itself can represent.
lmax      = 8;
NEXIobj   = NEXIrotinv(bval, Delta);
Nbval     = numel(bval);
Nb0       = Ngdir/16;

S_SH_NEXI = zeros(Nbval, Ngdir, Nsample);
theta     = acos(bvec(:,3));  % polar angle of each gradient direction vs. z

% per-voxel: Watson SH coefficients for this voxel's dispersion (kappa) ->
% NEXI signal SH coefficients for this voxel's (fa,Da,De,ra) -> convolve
% the two and project onto the actual sampled gradient directions (theta)
parfor k = 1:Nsample
    kappa       = pars(5,k);
    pl_NEXI     = NEXIobj.WatsonSH(kappa, lmax);
    F           = NEXIobj.NEXIsh(pars(1,k), pars(2,k), pars(3,k), pars(4,k), lmax);
    Si          = NEXIobj.SHconv(F, pl_NEXI, theta);
    S_SH_NEXI(:,:,k) = squeeze(Si);
end

% add b0 (1 per 16 DWI, appended as extra "directions" with signal = 1)
S_SH_NEXI(:, end+1:end+Nb0, :) = 1;

% add complex Gaussian noise, then take the real part (Rician-like
% magnitude bias is NOT modelled here - this is a simple additive-noise
% approximation, unlike e.g. gpuNEXIrice)
noise       = (1/SNR)*randn(size(S_SH_NEXI)) + 1i*(1/SNR)*randn(size(S_SH_NEXI));
S_noisy     = real(S_SH_NEXI + noise);

% --- add 1% background noise voxels ---
% Pure noise, no NEXI signal at all - stands in for CSF/outside-brain
% voxels a real mask wouldn't perfectly exclude. No ground truth exists
% for these; they're only used for the signal-vs-background loss
% diagnostics in sections 4/6/8/10, never for RMSE/scatter (sections
% 5/7/9, which use idx_signal only).
Nsample_sig = Nsample;
Nsample_bg  = round(0.01 * Nsample);
S_bg        = (1/SNR) * randn(Nbval, Ngdir+Nb0, Nsample_bg);

% combine signal and background (signal voxels first, background last -
% idx_signal/idx_bg below index into this same ordering throughout)
S_all       = cat(3, S_noisy, S_bg);
Nsample_all = Nsample_sig + Nsample_bg;
idx_signal  = 1:Nsample_sig;
idx_bg      = Nsample_sig+1:Nsample_all;

% reshape [measurement, direction, voxel] into GACELLE's expected
% [x,y,z,measurement] image-like layout (voxels placed along y, x/z
% singleton - same convention the NoisePropagation demos use)
S_all = permute(utils.vectorise_NDto2D(permute(S_all, [3 4 5 2 1])), [1 3 4 2]);
mask  = ones(size(S_all, 1:3), 'logical');

% per-measurement protocol arrays (bval/bvec/Delta/delta), each expanded
% to match S_all's flattened [gradient-direction-then-b0, shell] ordering
% - passed to gpuNEXI via extraData since 'data' is now the full
% acquisition rather than a pre-averaged rotationally-invariant signal
bval_all  = repmat(bval(:).', Ngdir,    1);
bval_all(end+1:end+Nb0, :) = 0;
bval_all  = bval_all(:);

bvec_all  = bvec;
bvec_all(end+1:end+Nb0, :) = 0;
bvec_all  = repmat(bvec_all, numel(bval), 1);

DELTA_all = repmat(Delta(:).', Ngdir+Nb0, 1);
DELTA_all = DELTA_all(:);

delta_all = repmat(delta(:).', Ngdir+Nb0, 1);
delta_all = delta_all(:);

% extraData struct
extraData.bval   = bval_all.';
extraData.bvec   = bvec_all.';
extraData.ldelta = delta_all.';
extraData.BDELTA = DELTA_all.';

% ground truth. p2 (the l=2 SH coefficient GACELLE actually fits) is
% recomputed here via NEXI's exact closed-form Watson SH expansion
% (NEXI.WatsonSHexact) rather than reusing NEXIrotinv.WatsonSH's
% lmax-truncated approximation from the forward simulation above, so
% "ground truth" means the true analytic value, not just whatever the
% simulator happened to use.
pl      = NEXI.WatsonSHexact(pars(5,:));
GT.fa   = pars(1,:);
GT.Da   = pars(2,:);
GT.De   = pars(3,:);
GT.ra   = pars(4,:);
GT.p2   = pl(2,:);
GT.tex  = (1-GT.fa) ./ GT.ra;

fprintf('Signal voxels     : %d\n', Nsample_sig);
fprintf('Background voxels : %d (%.1f%%)\n', Nsample_bg, 100*Nsample_bg/Nsample_all);
fprintf('Total voxels      : %d\n', Nsample_all);

%% ========================================================
% 2. Base fitting settings
% ========================================================
% Shared across all 6 configs below - only the convergence-related fields
% differ per config (section 3). lmax=2 here is deliberately lower than
% the lmax=8 used to forward-simulate the signal above: fitting only
% recovers the l=0/l=2 SH coefficients, so the higher-order terms in the
% simulated signal act as an (unmodelled) source of realism/noise, same
% as a real acquisition would have.
objGPU = gpuNEXI(bval, Delta);

fitting_base                    = objGPU.check_set_default([]);
fitting_base.iteration          = 4000;
fitting_base.initialLearnRate   = 0.001;
fitting_base.convergenceValue   = 1e-8;
fitting_base.lossFunction       = 'l1';
fitting_base.tol                = 1e-8;
fitting_base.isDisplay          = false;
fitting_base.lmax               = 2;
fitting_base.patience           = 5;
fitting_base.start              = 'likelihood';

%% ========================================================
% 3. Run all configurations
% ========================================================
% Same 6 configurations as demo_convergence_usage.m (see that script or
% docs/advanced/askadam_convergence.rst for a full explanation of each
% option) - repeated here against the harder, full-protocol + background-
% voxel dataset built above rather than the pre-averaged signal used
% there. rng/gpurng are reseeded before every call so all 6 configs (and
% the 0-iteration reference) start from the exact same random init and
% differences in the results are attributable only to the convergence
% settings, not to RNG drift between calls.

% --- Config 1: Original (linear, no robust) ---
fprintf('\n=== Config 1: Original (linear) ===\n');
rng(seed); gpurng(seed);
fitting                  = fitting_base;
fitting.convergenceModel = 'linear';
out_linear               = objGPU.estimate(S_all, mask, extraData, fitting);

% --- Config 2: EMA convergence model only ---
fprintf('\n=== Config 2: EMA convergence model ===\n');
rng(seed); gpurng(seed);
fitting                  = fitting_base;
fitting.convergenceModel = 'ema';
fitting.emaDecay         = 0.95;
out_ema                  = objGPU.estimate(S_all, mask, extraData, fitting);

% --- Config 3: Robust convergence + linear ---
% This is the config the 1% background voxels are specifically meant to
% stress-test: fitting.robustConvergence down-weights per-voxel outlier
% loss contributions (the outlier* fields tune how aggressively/quickly a
% voxel gets flagged and how much its loss is discounted) so that a
% minority of un-fittable voxels (here, pure-noise background) don't
% stall or bias convergence on the majority signal voxels. Section 6
% checks whether that's actually what happens.
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
out_robust                      = objGPU.estimate(S_all, mask, extraData, fitting);

% --- Config 4: Robust convergence + EMA ---
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
out_robust_ema                  = objGPU.estimate(S_all, mask, extraData, fitting);

% --- Config 5: Step norm signal only ---
% fitting.convergenceStepTol/.patienceStep stop iterating once the
% parameter-update step norm itself has stayed below tolerance for
% patienceStep iterations - a convergence signal independent of the loss
% value, useful when loss keeps drifting slightly (e.g. from the noisy
% background voxels) even though the parameters themselves have settled.
fprintf('\n=== Config 5: Step norm convergence signal ===\n');
rng(seed); gpurng(seed);
fitting                    = fitting_base;
fitting.convergenceModel   = 'linear';
fitting.convergenceStepTol = 1e-6;
fitting.patienceStep       = 5;
out_step                   = objGPU.estimate(S_all, mask, extraData, fitting);

% --- Config 6: All signals combined ---
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
out_all                         = objGPU.estimate(S_all, mask, extraData, fitting);

% --- Reference: starting point (0 iterations) ---
% fitting.iteration = 0 makes estimate() return the initial parameter
% guess untouched by any gradient step - a "Start" baseline in the
% comparisons below (sections 4/5/7/9) showing how much each config's
% actual fitting improved on the naive initialisation.
fprintf('\n=== Reference: starting point ===\n');
rng(seed); gpurng(seed);
fitting_ref           = fitting_base;
fitting_ref.iteration = 0;
pars0                 = objGPU.estimate(S_all, mask, extraData, fitting_ref);

% collect outputs
config_labels = {'Start', 'linear', 'EMA', 'robust+linear', 'robust+EMA', 'step', 'all'};
outs          = {out_linear, out_ema, out_robust, out_robust_ema, out_step, out_all};
all_outs      = [{pars0}, outs];   % Start + all 6 configs, for the sections that also compare against the naive initial guess
field         = fieldnames(GT);
field(strcmp(field,'tex')) = [];   % tex is derived, handled separately

%% ========================================================
% 4. Summary table
% ========================================================
% Loss (all) is estimate()'s own reported final loss (all voxels, signal
% + background); Loss (sig)/(bg) are the same per-voxel residual loss
% (final.resloss) averaged separately over the two voxel populations, to
% see how much of the overall loss is actually coming from the
% un-fittable background voxels rather than genuine signal misfit.
fprintf('\n=== Summary ===\n');
fprintf('%-30s | %10s | %12s | %12s | %12s\n', ...
    'Config', 'Iterations', 'Loss (all)', 'Loss (sig)', 'Loss (bg)');
fprintf('%s\n', repmat('-', 1, 85));
for k = 1:numel(outs)
    loss_sig = mean(outs{k}.final.resloss(idx_signal));
    loss_bg  = mean(outs{k}.final.resloss(idx_bg));
    fprintf('%-30s | %10d | %12.6e | %12.6e | %12.6e\n', ...
        config_labels{k+1}, outs{k}.final.Niteration, ...
        outs{k}.final.loss, loss_sig, loss_bg);
end

%% ========================================================
% 5. RMSE on signal voxels only
% ========================================================
% Unlike section 4's loss (which is a fit-quality proxy with no access to
% ground truth), this is the actual accuracy metric - and it's
% deliberately restricted to idx_signal throughout: the background
% voxels have no ground truth parameters to compare against (they were
% never generated from a NEXI signal in the first place).
fprintf('\n=== RMSE on signal voxels only ===\n');
fprintf('%-30s', 'Config');
for f = 1:numel(field); fprintf(' | %8s', field{f}); end
fprintf(' | %8s\n', 'tex');
fprintf('%s\n', repmat('-', 1, 30 + 11*(numel(field)+1)));

for k = 1:numel(outs)
    fprintf('%-30s', config_labels{k+1});
    for f = 1:numel(field)
        fitted = outs{k}.final.(field{f})(idx_signal);
        rmse   = sqrt(mean((fitted(:) - GT.(field{f})(:)).^2));
        fprintf(' | %8.4f', rmse);
    end
    tex_fitted = (1-outs{k}.final.fa(idx_signal)) ./ outs{k}.final.ra(idx_signal);
    tex_rmse   = sqrt(mean((tex_fitted(:) - GT.tex(:)).^2));
    fprintf(' | %8.4f\n', tex_rmse);
end

%% ========================================================
% 6. Background voxel diagnostic
% ========================================================
% The key check motivating this whole script: does robust convergence
% (Configs 3/4/6) help by genuinely fitting the signal voxels better, or
% just by learning to ignore/downweight the background ones? A high
% bg/signal loss ratio for the robust configs (vs. the non-robust ones)
% is the expected/healthy signature - it means the model is no longer
% spending effort trying to explain pure noise as if it were NEXI signal.
fprintf('\n=== Background voxel loss ratio (bg/signal) ===\n');
for k = 1:numel(outs)
    loss_bg  = mean(outs{k}.final.resloss(idx_bg));
    loss_sig = mean(outs{k}.final.resloss(idx_signal));
    fprintf('%-30s | BG: %.4e | Signal: %.4e | Ratio: %.1fx\n', ...
        config_labels{k+1}, loss_bg, loss_sig, loss_bg/loss_sig);
end

%% ========================================================
% 7. Plot: scatter fitted vs GT — signal voxels only
% ========================================================
% Grid: one row per parameter (fa, Da, De, ra, p2, plus the derived tex),
% one column per config including the "Start" reference - only
% idx_signal voxels are plotted since background voxels have no GT to
% compare against. The black reference line (refline(1), slope 1
% intercept 0) is the "fitted == ground truth" identity line, not a fit
% to the data.
all_field = [field; {'tex'}];
Nfield    = numel(all_field);

figure('Name', 'Fitted vs GT (signal voxels only)');
tiledlayout(Nfield, numel(all_outs), 'TileSpacing', 'compact');

for f = 1:numel(field)
    for k = 1:numel(all_outs)
        nexttile;
        scatter(GT.(field{f}), all_outs{k}.final.(field{f})(idx_signal), ...
            3, 'filled', 'MarkerFaceAlpha', 0.3);
        hold on; h = refline(1); h.Color = 'k';
        if f == 1; title(config_labels{k}); end
        if k == 1; ylabel(field{f}); end
        axis tight;
    end
end

% tex row
for k = 1:numel(all_outs)
    nexttile;
    tex_fitted = (1-all_outs{k}.final.fa(idx_signal)) ./ all_outs{k}.final.ra(idx_signal);
    scatter(GT.tex, tex_fitted, 3, 'filled', 'MarkerFaceAlpha', 0.3);
    hold on; h = refline(1); h.Color = 'k';
    if k == 1; ylabel('tex'); end
    axis tight;
end

%% ========================================================
% 8. Plot: per-voxel loss histogram — signal vs background
% ========================================================
% Two rows (signal / background) per config, sharing a common x-axis
% range (xmax, taken from Config 1) so the histograms are visually
% comparable across configs and across the signal/background split -
% the qualitative companion to section 6's numeric ratio.
figure('Name', 'Per-voxel loss: signal vs background');
tiledlayout(2, numel(outs), 'TileSpacing', 'compact');

xmax = max(outs{1}.final.resloss);

for k = 1:numel(outs)
    nexttile;
    histogram(outs{k}.final.resloss(idx_signal), 50, ...
        'FaceColor', [0.2 0.4 0.8], 'FaceAlpha', 0.6);
    xlabel('Loss'); ylabel('Count');
    title(sprintf('%s (signal)', config_labels{k+1}));
    xlim([0 xmax]);
end

for k = 1:numel(outs)
    nexttile;
    histogram(outs{k}.final.resloss(idx_bg), 10, ...
        'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.6);
    xlabel('Loss'); ylabel('Count');
    title(sprintf('%s (bg)', config_labels{k+1}));
    xlim([0 xmax]);
end

%% ========================================================
% 9. Plot: RMSE bar chart per parameter
% ========================================================
% One tile per parameter, bars over the 6 configs (Start excluded here -
% RMSE, unlike the scatter plots in section 7, is more useful compared
% config-to-config than against the trivial initial guess), with a
% dashed reference line at Config 1's ("linear") RMSE so any improvement
% or regression from the other 5 convergence settings is visible at a
% glance relative to that baseline.
figure('Name', 'RMSE per parameter (signal voxels only)');
tiledlayout(1, Nfield, 'TileSpacing', 'compact');

for f = 1:Nfield
    nexttile; hold on;
    rmse_vals = zeros(1, numel(outs));
    for k = 1:numel(outs)
        if strcmp(all_field{f}, 'tex')
            fitted = (1-outs{k}.final.fa(idx_signal)) ./ outs{k}.final.ra(idx_signal);
            gt_val = GT.tex(:);
        else
            fitted = outs{k}.final.(all_field{f})(idx_signal);
            gt_val = GT.(all_field{f})(:);
        end
        rmse_vals(k) = sqrt(mean((fitted(:) - gt_val).^2));
    end
    bar(rmse_vals);
    set(gca, 'XTick', 1:numel(outs), 'XTickLabel', config_labels(2:end), ...
        'XTickLabelRotation', 30);
    ylabel('RMSE'); title(all_field{f});
    yline(rmse_vals(1), '--k', 'Baseline');
end

%% ========================================================
% 10. Plot: per-sample loss with background boundary marker
% ========================================================
% Every voxel's loss plotted against its index in S_all's fixed ordering
% (signal voxels first, background last - see the "combine signal and
% background" comment in section 1), with a dashed vertical line at the
% signal/background boundary. The 1% background voxels sit at the far
% right of every panel; whether their loss visually separates from the
% signal cluster (rather than blending into it) is the same
% robust-convergence question section 6 answers numerically.
figure('Name', 'Per-sample residual loss');
tiledlayout(1, 1); nexttile; hold on;
colors  = lines(numel(outs)+1);
markers = {'x','o','+','s','d','^','v'};

scatter(1:Nsample_all, pars0.final.resloss, 5, colors(1,:), markers{1}, ...
    'DisplayName', 'Start');
for k = 1:numel(outs)
    scatter(1:Nsample_all, outs{k}.final.resloss, 5, colors(k+1,:), markers{k+1}, ...
        'MarkerFaceAlpha', 0.4, 'MarkerEdgeAlpha', 0.4, ...
        'DisplayName', config_labels{k+1});
end
xline(Nsample_sig + 0.5, '--k', 'BG boundary', 'LabelVerticalAlignment', 'bottom');
legend('Location', 'best');
xlabel('Sample index'); ylabel('Loss');
title('Per-sample residual loss (right of dashed line = background voxels)');