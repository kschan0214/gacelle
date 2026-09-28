%% test_T2_2_nuisance_recovery.m
%
% T2.2 (Phase 2): nuisance recovery under 'marginal_S0noise'. The sampler
% returns sigma ('noise') and S0 by post-hoc exact conditional draws for every
% retained sample; their marginal posteriors must match a reference.
%
% Cases (same data and sampler settings as T2.1)
%   (a) monoexponential (R2star, S0 = M0, sigma), 20 voxels, SNR 20 and 50.
%       Reference: brute-force 3D grid over (R2star, S0, t = log sigma^2) of the
%       JOINT posterior, written from the priors only (no analytic marginal or
%       conditional is used):
%         p(R2star) flat on [0.1,200]; p(S0|sigma^2,R2star) = Zellner g-prior
%         N(0, k sigma^2/c) in the broad limit, ∝ sqrt(c)/sigma, c = g'g;
%         p(sigma^2) ∝ 1/sigma^2; Gaussian likelihood; d sigma^2 = sigma^2 dt:
%         log p = 0.5 log c - (m+1)/2 t - Q exp(-t)/2,  Q = |y - S0 g|^2.
%       Coarse 200^3 grid over R2star [0.1,200] x S0 [0,2] x t [-20,0], then
%       150^3 over the bounding box of log p > max - 25 (spacing ~0.1 marginal
%       SD); check grid 100^3 for eGrid = |q_150 - q_100|. sigma quantiles are
%       exp(q_t/2), density f_sigma = f_t * 2/sigma.
%   (b) IVIM (D, F, Dstar, S0, sigma), 5 voxels, SNR 20 and 50. Reference: the
%       refined 150^3 grid posterior of (D, F, Dstar) from T2.1, integrated
%       against the analytic conditionals by Monte Carlo: Nref = 4e6 grid nodes
%       drawn with probability ∝ trapezoid weight x posterior, then
%       sigma^2 ~ InvGamma(m/2, RSS/2), S0 ~ N(Shat, sigma^2/c) (double, GPU).
%       This part uses the same conditional formulas as the sampler, so it
%       checks the implementation (state caching, thinning, draws), while (a)
%       also checks the formulas. eRef = sqrt(q(1-q)/Nref)/f_ref (iid), f_ref by
%       a central difference of the empirical CDF (h = 0.05 SD).
%   Transforms 'linear' and 'sigmoid'.
%
% Test statistic: z = (q_mcmc - q_ref)/sqrt(se^2 + eRef^2) per voxel,
%   parameter (noise, S0) and quantile in {2.5,25,50,75,97.5}% (quantile_z.m,
%   se from the ESS of the indicator and the reference density).
% Criterion (stated before running), per case, pooled over SNR x transform
%   ((a) 20*2*5*2*2 = 800 z, (b) 5*2*5*2*2 = 200 z):
%   PASS if frac(|z| > 1.96) <= 0.10 AND max|z| <= 4.5.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T2_2_nuisance_recovery
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
Nref        = 4e6;

% same seeds as T2.1 (same data, starts and runs)
seedDataA   = [31 32];
seedDataB   = [41 42];
seedStart   = 51;
seedRun     = [61 62];
seedRef     = 71;

fprintf('T2.2 nuisance recovery, marginal_S0noise. SNR %s, transforms %s, %d chains\n', mat2str(SNRs), strjoin(transforms,'/'), Nchain);
fprintf('Seeds: data (a) %s (b) %s, start %d, run %s, reference draws %d\n', mat2str(seedDataA), mat2str(seedDataB), seedStart, mat2str(seedRun), seedRef);
fprintf('Criterion per case: frac(|z|>1.96) <= %.2f and max|z| <= %.1f (pooled over SNR x transform)\n\n', zFracMax, zMaxMax);

%% (a) monoexponential, 3D joint grid
NvA = 20;
zA  = [];
fprintf('=== (a) monoexponential: 3D joint grid over (R2star, S0, log sigma^2), %d voxels ===\n', NvA);
for ks = 1:numel(SNRs)
    SNR = SNRs(ks);
    [y4, ~, ~, fitting, obj, truth] = r2star_sim(NvA, SNR, seedDataA(ks));
    y   = double(reshape(y4, NvA, []));
    te  = double(obj.te(:));
    m   = numel(te);
    iR  = find(strcmp(fitting.modelParams,'R2star'));
    ranges = [fitting.lb(iR) fitting.ub(iR); 0 2; -20 0];

    t0 = tic;
    qRef = struct('noise', zeros(NvA,numel(qs)), 'M0', zeros(NvA,numel(qs)));
    fRef = qRef; eRef = qRef; edgeM = zeros(NvA,1); hSD = zeros(NvA,3);
    for v = 1:NvA
        lpf = @(A,B,C) logjoint_mono(A, B, C, te, y(v,:), m);
        G   = grid3_refine(lpf, ranges, 150, 200, 25, 100);
        edgeM(v) = G.edgeMass;
        % S0 (axis 2)
        [q, f, sd]  = grid_quantiles(G.axes{2}, G.marg{2}, qs);
        qc          = grid_quantiles(G.axesCheck{2}, G.margCheck{2}, qs);
        qRef.M0(v,:) = q; fRef.M0(v,:) = f; eRef.M0(v,:) = abs(q - qc); hSD(v,2) = (G.axes{2}(2)-G.axes{2}(1))/sd;
        % sigma = exp(t/2) (axis 3)
        [q, f, sd]  = grid_quantiles(G.axes{3}, G.marg{3}, qs);
        qc          = grid_quantiles(G.axesCheck{3}, G.margCheck{3}, qs);
        s           = exp(q/2);
        qRef.noise(v,:) = s; fRef.noise(v,:) = f .* 2 ./ s; eRef.noise(v,:) = abs(s - exp(qc/2)); hSD(v,3) = (G.axes{3}(2)-G.axes{3}(1))/sd;
        [~, ~, sd]  = grid_quantiles(G.axes{1}, G.marg{1}, qs); hSD(v,1) = (G.axes{1}(2)-G.axes{1}(1))/sd;
    end
    fprintf('SNR %d: grid %.1f s; spacing/SD max [R2star S0 t] = %s; max edge mass %.1e\n', SNR, toc(t0), mat2str(max(hSD,[],1),2), max(edgeM));
    fprintf('  grid median sigma / truth: median %.3f; grid median S0 - truth: median %.4f\n', ...
        median(qRef.noise(:,3)./truth.noise(:)), median(qRef.M0(:,3) - truth.M0(:)));

    for kt = 1:numel(transforms)
        f = fitting;
        f.likelihood = 'marginal_S0noise'; f.S0Param = 'M0';
        f.parameterTransform = transforms{kt};
        f.updateScheme = 'joint'; f.adaptStepSize = true; f.adaptInterval = 50;
        f.iteration = 40000; f.burnin = 10000; f.thinning = 5;
        f.metric = {'mean'};
        x0 = random_starts(f, NvA*Nchain, seedStart + ks);
        [post, ~, tRun] = run_chains(y, f, @(p) obj.FWD(p, 'mcmc'), x0, Nchain, seedRun(kt));
        msg = sprintf('  %-7s: run %5.1f s', transforms{kt}, tRun);
        for p = {'noise','M0'}
            z = zeros(NvA, numel(qs));
            for v = 1:NvA
                z(v,:) = quantile_z(post.(p{1})(v,:,:), qRef.(p{1})(v,:), fRef.(p{1})(v,:), eRef.(p{1})(v,:), qs);
            end
            zA = [zA; z(:)]; %#ok<AGROW>
            msg = [msg sprintf(' | %s: max|z| %.2f, frac|z|>1.96 %.3f, R-hat max %.4f', p{1}, max(abs(z(:))), mean(abs(z(:))>1.96), max(mcmc_bayes.rhat(post.(p{1}))))]; %#ok<AGROW>
        end
        fprintf('%s\n', msg);
    end
end
passA = mean(abs(zA) > 1.96) <= zFracMax && max(abs(zA)) <= zMaxMax;
fprintf('(a) pooled: n=%d, frac|z|>1.96 = %.3f, max|z| = %.2f -> %s\n\n', numel(zA), mean(abs(zA)>1.96), max(abs(zA)), pf(passA));

%% (b) IVIM, grid posterior of (D,F,Dstar) x analytic conditionals
cfg  = ivim_t2_config();
NvB  = numel(cfg.truth.D);
m    = numel(cfg.b);
zB   = [];
iNL  = find(ismember(cfg.modelParams, cfg.nonlinear));
box  = [cfg.lb(iNL) cfg.ub(iNL)];
fprintf('=== (b) IVIM: grid posterior x exact conditionals (Nref = %.0e), %d voxels ===\n', Nref, NvB);
for ks = 1:numel(SNRs)
    SNR = SNRs(ks);
    y   = ivim_sim(cfg, SNR, 1, seedDataB(ks), 'gaussian');

    t0 = tic;
    qRef = struct('noise', zeros(NvB,numel(qs)), 'S0', zeros(NvB,numel(qs)));
    fRef = qRef; eRef = qRef;
    for v = 1:NvB
        lpf = @(A,B,C) logpost_ivim(A, B, C, cfg.b, y(v,:), m);
        G   = grid3_refine(lpf, box, 150, 150, 25, 100);
        [s, S0] = reference_draws(G, cfg.b, y(v,:), m, Nref, seedRef + 100*ks + v);
        [qRef.noise(v,:), fRef.noise(v,:), eRef.noise(v,:)] = empirical_ref(s,  qs);
        [qRef.S0(v,:),    fRef.S0(v,:),    eRef.S0(v,:)]    = empirical_ref(S0, qs);
    end
    fprintf('SNR %d: grid + reference draws %.1f s; ref median sigma/truth %s; ref median S0 - truth %s\n', SNR, toc(t0), ...
        mat2str(qRef.noise(:,3).'./(cfg.truth.S0/SNR), 3), mat2str(qRef.S0(:,3).' - cfg.truth.S0, 3));

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
        msg = sprintf('  %-7s: run %5.1f s', transforms{kt}, tRun);
        for p = {'noise','S0'}
            z = zeros(NvB, numel(qs));
            for v = 1:NvB
                z(v,:) = quantile_z(post.(p{1})(v,:,:), qRef.(p{1})(v,:), fRef.(p{1})(v,:), eRef.(p{1})(v,:), qs);
            end
            zB = [zB; z(:)]; %#ok<AGROW>
            msg = [msg sprintf(' | %s: max|z| %.2f, frac|z|>1.96 %.3f, R-hat max %.4f', p{1}, max(abs(z(:))), mean(abs(z(:))>1.96), max(mcmc_bayes.rhat(post.(p{1}))))]; %#ok<AGROW>
        end
        fprintf('%s\n', msg);
    end
end
passB = mean(abs(zB) > 1.96) <= zFracMax && max(abs(zB)) <= zMaxMax;
fprintf('(b) pooled: n=%d, frac|z|>1.96 = %.3f, max|z| = %.2f -> %s\n\n', numel(zB), mean(abs(zB)>1.96), max(abs(zB)), pf(passB));

fprintf('T2.2 overall: %s   (total time %.1f s)\n', pf(passA && passB), toc(tStart));

%% local functions
function lp = logjoint_mono(R2, S0, t, te, y, m)
% joint log posterior of (R2star, S0, t = log sigma^2), from the priors only
te4 = reshape(gpuArray(te), 1, 1, 1, []);
y4  = reshape(gpuArray(double(y)), 1, 1, 1, []);
g   = exp(-te4 .* R2);                      % [nA,1,1,m]
c   = sum(g.^2, 4);                         % [nA,1,1]
yg  = sum(y4.*g, 4);
yy  = sum(y4.^2, 4);
Q   = yy - 2*S0.*yg + S0.^2.*c;             % [nA,nB,1]
lp  = 0.5*log(c) - (m+1)/2*t - Q.*exp(-t)/2;
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

function [s, S0] = reference_draws(G, b, y, m, N, seed)
% theta nodes ~ grid posterior (trapezoid weights), then exact conditionals (double, GPU)
rng(seed); parallel.gpu.rng(seed);
p       = G.w(:) .* exp(G.logp(:));
cp      = [0; cumsum(p)/sum(p)];
cnt     = histcounts(rand(N,1), cp);
idx     = repelem((1:numel(p)).', cnt(:));
[i1,i2,i3] = ind2sub(size(G.logp), idx);
D       = G.axes{1}(i1); F = G.axes{2}(i2); Ds = G.axes{3}(i3);
s = zeros(N,1); S0 = zeros(N,1);
b = gpuArray(double(b(:))); yv = gpuArray(double(y(:)));
chunk = 5e5;
for k = 1:chunk:N
    j   = k:min(k+chunk-1, N);
    Dj  = gpuArray(D(j).'); Fj = gpuArray(F(j).'); Dsj = gpuArray(Ds(j).');
    g   = Fj.*exp(-b.*Dsj) + (1-Fj).*exp(-b.*Dj);      % [m, n]
    c   = sum(g.^2, 1);
    Sh  = sum(yv.*g, 1)./c;
    RSS = sum((yv - Sh.*g).^2, 1);
    s2  = (RSS/2) ./ randg(m/2, size(RSS), 'like', RSS);
    s(j)  = gather(sqrt(s2));
    S0(j) = gather(Sh + sqrt(s2./c).*randn(size(Sh), 'like', Sh));
end
end

function [q, f, e] = empirical_ref(x, qs)
% quantiles, density (central difference of the empirical CDF, h = 0.05 SD) and iid SE
xs = sort(x); N = numel(xs);
q  = xs(max(1, ceil(qs*N))).';
h  = 0.05*std(x);
f  = zeros(size(qs));
for k = 1:numel(qs)
    f(k) = (sum(xs <= q(k)+h) - sum(xs <= q(k)-h)) / (2*h*N);
end
e  = sqrt(qs.*(1-qs)/N) ./ f;
end

function x0 = random_starts(f, N, seed)
rng(seed);
for k = 1:numel(f.modelParams)
    x0.(f.modelParams{k}) = f.lb(k) + (f.ub(k)-f.lb(k)) * (0.1 + 0.8*rand(N,1));
end
end

function s = pf(c)
if c; s = 'PASS'; else; s = 'FAIL'; end
end
