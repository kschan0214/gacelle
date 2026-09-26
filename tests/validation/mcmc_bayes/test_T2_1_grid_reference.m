%% test_T2_1_grid_reference.m
%
% T2.1 (Phase 2): marginal likelihood 'marginal_S0noise' (Zellner g-prior on
% S0 in the broad limit + p(sigma^2) ∝ 1/sigma^2) against a brute-force grid
% reference under the same prior (flat in the native box of the non-linear
% parameters). Sampler: mcmc_bayes on the GPU (single precision). Grid: double
% precision on the GPU.
%
% Cases
%   (a) monoexponential, S0 (M0) marginalised, 1 non-linear parameter (R2star):
%       gpuR2starMapping FWD, 20 voxels (r2star_sim.m: M0 ~ U(0.9,1.1),
%       R2star ~ U(20,50) 1/s, 12 echoes 0-40 ms), box R2star in [0.1, 200].
%       Grid: 100001 nodes over the whole box (spacing 0.002 1/s, i.e. < 0.002
%       posterior SD at SNR 50); check grid 50001 nodes.
%   (b) IVIM (ivim_fwd.m; D, F, Dstar), S0 marginalised, 5 voxels
%       (ivim_t2_config.m), b = 0..900 s/mm^2 (16 values of the draft plan, in
%       ms/um^2). Box D [0.1,3], F [0,0.5], Dstar [4,100] (um^2/ms).
%       Grid: coarse 150^3 over the box, then 150^3 over the bounding box of
%       the coarse nodes with log posterior > max - 25 (+2 coarse nodes), so the
%       refined spacing is ~0.1 marginal SD (14 SD / 150) for a Gaussian-like
%       posterior, and up to ~0.3 SD for the skewed/truncated IVIM posteriors
%       here (printed at run time); the trapezoid rule then resolves the
%       marginals well below MC error. A second refined grid with 100^3 nodes over the same box gives
%       an empirical discretisation error eGrid = |q_150 - q_100| (conservative:
%       it is the error of the coarser grid).
%   SNR = 20 and 50 (sigma = S0/SNR), Gaussian noise, unit weights.
%   Each case is run with parameterTransform 'linear' and 'sigmoid' (checks the
%   Jacobian handling under the marginal likelihood).
%
% Sampler (both cases): joint updates, adaptStepSize (burn-in only), 4
%   independent chains per voxel (run_chains.m: 4 copies of every voxel in one
%   call), start points drawn uniformly in the central 80% of the box, i.e.
%   over-dispersed. (a) 40000 iterations, burn-in 10000, thinning 5;
%   (b) 100000 iterations, burn-in 20000, thinning 10.
%
% Test statistic, per voxel, parameter and quantile q in {2.5,25,50,75,97.5}%:
%   z = (q_mcmc - q_grid) / sqrt(se^2 + eGrid^2),
%   se = sqrt(q(1-q)/ESS_q) / f_grid(q_grid)   (quantile_z.m; ESS_q = multi-chain
%   ESS of the indicator 1{x <= q_grid}, f_grid the grid marginal density).
% Criterion (stated before running), per case (a) and (b), pooled over SNR and
%   transform ((a): 20*1*5*2*2 = 400 z, (b): 5*3*5*2*2 = 300 z):
%   PASS if frac(|z| > 1.96) <= 0.10 AND max|z| <= 4.5.
%   (Nominal 0.05; slack for correlated quantiles within a voxel and ESS-estimate
%   noise. Under independence P(Bin(300,0.05) > 30) ~ 0.1%.)
%   Split-R-hat (4 chains) is reported per parameter; R-hat > 1.05 is flagged as
%   a mixing problem (reported, the z criterion decides PASS/FAIL).
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T2_1_grid_reference
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%

clearvars; tStart = tic;

qs          = [0.025 0.25 0.5 0.75 0.975];
SNRs        = [20 50];
transforms  = {'linear','sigmoid'};
Nchain      = 4;
zFracMax    = 0.10;
zMaxMax     = 4.5;
rhatFlag    = 1.05;

seedDataA   = [31 32];      % per SNR
seedDataB   = [41 42];
seedStart   = 51;
seedRun     = [61 62];      % per transform

fprintf('T2.1 grid reference, marginal_S0noise. SNR %s, transforms %s, %d chains\n', mat2str(SNRs), strjoin(transforms,'/'), Nchain);
fprintf('Seeds: data (a) %s (b) %s, start %d, run %s\n', mat2str(seedDataA), mat2str(seedDataB), seedStart, mat2str(seedRun));
fprintf('Criterion per case: frac(|z|>1.96) <= %.2f and max|z| <= %.1f (pooled over SNR x transform)\n\n', zFracMax, zMaxMax);

%% (a) monoexponential
NvA = 20;
zA  = [];
fprintf('=== (a) monoexponential (R2star), %d voxels ===\n', NvA);
for ks = 1:numel(SNRs)
    SNR = SNRs(ks);
    [y4, ~, ~, fitting, obj] = r2star_sim(NvA, SNR, seedDataA(ks));
    y   = double(reshape(y4, NvA, []));                 % [Nv, Nte]
    te  = double(obj.te(:));
    m   = numel(te);

    % grid reference
    t0 = tic;
    iR  = find(strcmp(fitting.modelParams,'R2star'));
    x   = linspace(fitting.lb(iR), fitting.ub(iR), 100001);
    xc  = linspace(fitting.lb(iR), fitting.ub(iR), 50001);
    qG  = zeros(NvA, numel(qs)); fG = qG; eG = qG; sdG = zeros(NvA,1);
    for v = 1:NvA
        [qG(v,:), fG(v,:), sdG(v)] = grid_quantiles(x,  exp(logpost_mono(x,  te, y(v,:), m)), qs);
        qc                         = grid_quantiles(xc, exp(logpost_mono(xc, te, y(v,:), m)), qs);
        eG(v,:) = abs(qG(v,:) - qc);
    end
    fprintf('SNR %d: grid %.1f s, spacing/SD median %.1e, max eGrid/SD %.1e\n', SNR, toc(t0), ...
        median((x(2)-x(1))./sdG), max(eG./sdG,[],'all'));

    for kt = 1:numel(transforms)
        f = fitting;
        f.likelihood = 'marginal_S0noise'; f.S0Param = 'M0';
        f.parameterTransform = transforms{kt};
        f.updateScheme = 'joint'; f.adaptStepSize = true; f.adaptInterval = 50;
        f.iteration = 40000; f.burnin = 10000; f.thinning = 5;
        f.metric = {'mean'};
        x0 = random_starts(f, NvA*Nchain, seedStart + ks);
        [post, ~, tRun] = run_chains(y, f, @(p) obj.FWD(p, 'mcmc'), x0, Nchain, seedRun(kt));
        z = zeros(NvA, numel(qs));
        for v = 1:NvA
            z(v,:) = quantile_z(post.R2star(v,:,:), qG(v,:), fG(v,:), eG(v,:), qs);
        end
        R = mcmc_bayes.rhat(post.R2star);
        E = mcmc_bayes.ess(post.R2star);
        zA = [zA; z(:)]; %#ok<AGROW>
        fprintf('  %-7s: run %5.1f s | R2star: max|z| %.2f, frac|z|>1.96 %.3f, R-hat max %.4f%s, ESS median %.0f\n', ...
            transforms{kt}, tRun, max(abs(z(:))), mean(abs(z(:))>1.96), max(R), flag(max(R) > rhatFlag), median(E));
    end
end
passA = mean(abs(zA) > 1.96) <= zFracMax && max(abs(zA)) <= zMaxMax;
fprintf('(a) pooled: n=%d, frac|z|>1.96 = %.3f, max|z| = %.2f -> %s\n\n', numel(zA), mean(abs(zA)>1.96), max(abs(zA)), pf(passA));

%% (b) IVIM
cfg  = ivim_t2_config();
NvB  = numel(cfg.truth.D);
m    = numel(cfg.b);
zB   = [];
iNL  = find(ismember(cfg.modelParams, cfg.nonlinear));
box  = [cfg.lb(iNL) cfg.ub(iNL)];
fprintf('=== (b) IVIM (D, F, Dstar), %d voxels, %d b-values ===\n', NvB, m);
for ks = 1:numel(SNRs)
    SNR = SNRs(ks);
    y   = ivim_sim(cfg, SNR, 1, seedDataB(ks), 'gaussian');     % [Nv, Nb]

    % grid reference per voxel
    t0 = tic;
    qG = zeros(NvB, 3, numel(qs)); fG = qG; eG = qG; hSD = zeros(NvB,3); edgeM = zeros(NvB,1);
    for v = 1:NvB
        lpf = @(A,B,C) logpost_ivim(A, B, C, cfg.b, y(v,:), m);
        G   = grid3_refine(lpf, box, 150, 150, 25, 100);
        edgeM(v) = G.edgeMass;
        for kp = 1:3
            [qG(v,kp,:), fG(v,kp,:), sd] = grid_quantiles(G.axes{kp}, G.marg{kp}, qs);
            qc = grid_quantiles(G.axesCheck{kp}, G.margCheck{kp}, qs);
            eG(v,kp,:) = abs(squeeze(qG(v,kp,:)).' - qc);
            hSD(v,kp)  = (G.axes{kp}(2) - G.axes{kp}(1)) / sd;
        end
    end
    fprintf('SNR %d: grid %.1f s; spacing/SD max [D F Dstar] = %s; max edge mass %.1e\n', SNR, toc(t0), ...
        mat2str(max(hSD,[],1),2), max(edgeM));
    for v = 1:NvB
        fprintf('  voxel %d grid median [D F Dstar] = [%.3f %.4f %.2f], 95%% CI D [%.3f %.3f] F [%.4f %.4f] Dstar [%.1f %.1f]\n', v, ...
            qG(v,1,3), qG(v,2,3), qG(v,3,3), qG(v,1,1), qG(v,1,5), qG(v,2,1), qG(v,2,5), qG(v,3,1), qG(v,3,5));
    end

    for kt = 1:numel(transforms)
        f = struct();
        f.modelParams = cfg.modelParams; f.lb = cfg.lb; f.ub = cfg.ub; f.xStepSize = cfg.xStepSize;
        f.algorithm = 'MH';
        f.likelihood = 'marginal_S0noise'; f.S0Param = 'S0';
        f.parameterTransform = transforms{kt};
        f.updateScheme = 'joint'; f.adaptStepSize = true; f.adaptInterval = 50;
        f.iteration = 100000; f.burnin = 20000; f.thinning = 10;
        f.metric = {'mean'};
        x0 = random_starts(f, NvB*Nchain, seedStart + 10 + ks);
        [post, ~, tRun] = run_chains(y, f, @(p) ivim_fwd(p, cfg.b), x0, Nchain, seedRun(kt));
        fprintf('  %-7s: run %5.1f s\n', transforms{kt}, tRun);
        for kp = 1:3
            p = cfg.nonlinear{kp};
            z = zeros(NvB, numel(qs));
            for v = 1:NvB
                z(v,:) = quantile_z(post.(p)(v,:,:), squeeze(qG(v,kp,:)).', squeeze(fG(v,kp,:)).', squeeze(eG(v,kp,:)).', qs);
            end
            R = mcmc_bayes.rhat(post.(p));
            E = mcmc_bayes.ess(post.(p));
            zB = [zB; z(:)]; %#ok<AGROW>
            fprintf('    %-6s max|z| %.2f, frac|z|>1.96 %.3f, R-hat max %.4f%s, ESS min/median %.0f/%.0f | z by voxel (max|z|): %s\n', ...
                p, max(abs(z(:))), mean(abs(z(:))>1.96), max(R), flag(max(R) > rhatFlag), min(E), median(E), ...
                mat2str(max(abs(z),[],2).', 3));
        end
    end
end
passB = mean(abs(zB) > 1.96) <= zFracMax && max(abs(zB)) <= zMaxMax;
fprintf('(b) pooled: n=%d, frac|z|>1.96 = %.3f, max|z| = %.2f -> %s\n\n', numel(zB), mean(abs(zB)>1.96), max(abs(zB)), pf(passB));

fprintf('T2.1 overall: %s   (total time %.1f s)\n', pf(passA && passB), toc(tStart));

%% local functions
function lp = logpost_mono(x, te, y, m)
% -(m/2) log RSS(R2star), unit weights, grid in double on the GPU
x   = gpuArray(double(x(:).'));
g   = exp(-te .* x);                         % [m, n]
y   = gpuArray(double(y(:)));
c   = sum(g.^2, 1);
S   = sum(y.*g, 1) ./ c;
RSS = sum((y - S.*g).^2, 1);
lp  = gather(-(m/2)*log(RSS));
lp  = lp - max(lp);
end

function lp = logpost_ivim(D, F, Dstar, b, y, m)
% -(m/2) log RSS(D,F,Dstar) on a 3D node grid, unit weights (flat box prior)
b4  = reshape(gpuArray(double(b)), 1, 1, 1, []);
y4  = reshape(gpuArray(double(y)), 1, 1, 1, []);
g   = F.*exp(-b4.*Dstar) + (1-F).*exp(-b4.*D);
c   = sum(g.^2, 4);
S   = sum(y4.*g, 4) ./ c;
RSS = sum((y4 - S.*g).^2, 4);
lp  = -(m/2)*log(RSS);
end

function x0 = random_starts(f, N, seed)
% start points uniform in the central 80% of the box, one per chain copy
rng(seed);
for k = 1:numel(f.modelParams)
    x0.(f.modelParams{k}) = f.lb(k) + (f.ub(k)-f.lb(k)) * (0.1 + 0.8*rand(N,1));
end
end

function s = pf(c)
if c; s = 'PASS'; else; s = 'FAIL'; end
end

function s = flag(c)
if c; s = ' (NOT MIXED)'; else; s = ''; end
end
