addpath(genpath('/autofs/space/linen_001/users/kwokshing/tools/gacelle/'));
addpath(genpath('../../dwi/C2_protocoldesign'));
clear;

%% ========================================================
% 1. Simulate data
% ========================================================
seed = 23439; rng(seed); gpurng(seed);

SNR     = 50;
Ngdir   = 32;
Nsample = 1e3;

bval_unique  = [2.3 3.5 4.8 6.5 11.5 17.5];
delta_little = 6;
DELTA_big    = [13, 21, 30];
Nshell       = [4, 5, 6];
bval         = [bval_unique(1:Nshell(1)) ...
                bval_unique(1:Nshell(2)) ...
                bval_unique(1:Nshell(3))].';

method = 'matlab';
pd     = protocoldesign();
bvec   = pd.dirgen(Ngdir, method);

Delta = [DELTA_big(1)*ones(Nshell(1),1); ...
         DELTA_big(2)*ones(Nshell(2),1); ...
         DELTA_big(3)*ones(Nshell(3),1)];
delta = delta_little * ones(numel(bval), 1);

% ground truth parameters
intervals = [0.1  0.9  ; ...   % fa
             1.5  3.0  ; ...   % Da
             0.5  1.5  ; ...   % De
             2.0  50.0 ; ...   % tex -> converted to ra below
             0.5  1.5 ];      % kappa

pars      = intervals(:,1) + rand(size(intervals,1), Nsample) .* diff(intervals, [], 2);
pars(4,:) = max((1./pars(4,:)) .* (1-pars(1,:)), 1/200);  % tex -> ra

% enforce Da >= De
tmp      = pars;
ind      = find(pars(2,:) < pars(3,:));
pars(2,ind) = tmp(3,ind);
pars(3,ind) = tmp(2,ind);

% forward signal generation
lmax      = 8;
NEXIobj   = NEXIrotinv(bval, Delta);
Nbval     = numel(bval);
Nb0       = Ngdir/16;

S_SH_NEXI = zeros(Nbval, Ngdir, Nsample);
theta     = acos(bvec(:,3));

parfor k = 1:Nsample
    kappa       = pars(5,k);
    pl_NEXI     = NEXIobj.WatsonSH(kappa, lmax);
    F           = NEXIobj.NEXIsh(pars(1,k), pars(2,k), pars(3,k), pars(4,k), lmax);
    Si          = NEXIobj.SHconv(F, pl_NEXI, theta);
    S_SH_NEXI(:,:,k) = squeeze(Si);
end

% add b0 (1 per 16 DWI)
S_SH_NEXI(:, end+1:end+Nb0, :) = 1;

% add noise
noise       = (1/SNR)*randn(size(S_SH_NEXI)) + 1i*(1/SNR)*randn(size(S_SH_NEXI));
S_noisy     = real(S_SH_NEXI + noise);

% --- add 1% background noise voxels ---
Nsample_sig = Nsample;
Nsample_bg  = round(0.01 * Nsample);
S_bg        = (1/SNR) * randn(Nbval, Ngdir+Nb0, Nsample_bg);

% combine signal and background
S_all       = cat(3, S_noisy, S_bg);
Nsample_all = Nsample_sig + Nsample_bg;
idx_signal  = 1:Nsample_sig;
idx_bg      = Nsample_sig+1:Nsample_all;

% reshape to expected layout
S_all = permute(utils.vectorise_NDto2D(permute(S_all, [3 4 5 2 1])), [1 3 4 2]);
mask  = ones(size(S_all, 1:3), 'logical');

% protocol arrays
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

% ground truth
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
fprintf('\n=== Reference: starting point ===\n');
rng(seed); gpurng(seed);
fitting_ref           = fitting_base;
fitting_ref.iteration = 0;
pars0                 = objGPU.estimate(S_all, mask, extraData, fitting_ref);

% collect outputs
config_labels = {'Start', 'linear', 'EMA', 'robust+linear', 'robust+EMA', 'step', 'all'};
outs          = {out_linear, out_ema, out_robust, out_robust_ema, out_step, out_all};
all_outs      = [{pars0}, outs];
field         = fieldnames(GT);
field(strcmp(field,'tex')) = [];   % tex is derived, handled separately

%% ========================================================
% 4. Summary table
% ========================================================
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