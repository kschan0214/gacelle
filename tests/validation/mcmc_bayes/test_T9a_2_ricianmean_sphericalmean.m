%% test_T9a_2_ricianmean_sphericalmean.m
%
% T9a.2 (Phase 9a): characterisation, NO pass/fail.
%
% Spherical-mean phantom for gpumcmicro (SMT: stick with D plus zeppelin with D_par = D,
% D_perp = (1-f) D, single TE). The signal is simulated at the DIRECTION level: Ndir directions per
% shell (golden-spiral set, randomly rotated per shell and voxel), one random fibre direction per
% voxel, complex Gaussian noise (sigma_dir = 1/SNR per channel, S0 = 1 noise-free, i.e. the data are
% b0-normalised with a noise-free b0), magnitude per direction, then the direction average per shell.
% The noise-free directional signals of each shell are rescaled so that their average equals the
% class spherical mean exactly (removes the finite-direction quadrature error, up to ~3e-2 with 30
% directions at b = 8, so that only the noise model differs between data and model).
% At high b the averaged magnitude sits on the Rician noise floor. Compared (same data, same seeds):
%   'gaussian'                               Gaussian around the spherical-mean model, sigma sampled
%   'gaussian_ricianmean', ricianNav = Ndir  Gaussian around the Rician mean, sigma_s = sigma*sqrt(Ndir)
%   'gaussian_ricianmean', ricianSigma       Gaussian around the Rician mean, sigma_s = sigma_dir (known)
%
% Truth: f in {0.4, 0.6, 0.8} x D in {1.8, 2.4} um^2/ms (6 truth voxels), Nreal realisations each;
% b = [1 2.5 5 8] ms/um^2, Ndir = 30, SNR 5 and 10. One chain per voxel, parameterTransform sigmoid
% (f, D) and log (noise), joint updates, adaptStepSize, flat box prior (f in [1e-6, 1], D in [1e-6, 3],
% noise in [1e-3, 0.5]).
%
% Reported per SNR and likelihood:
%   high-b: mean over voxels of (data - noise-free spherical mean) at the highest b (the floor), and the
%     bias of the model prediction of the expected datum at the highest b, evaluated at the posterior
%     means: gaussian: FWD; ricianmean: E(FWD, sigma_s), minus the true expected datum
%     mean_dir E_Rice(S_dir, sigma_dir). Also the spherical-mean approximation error at the truth
%     E(mean_dir S_dir, sigma_dir) - mean_dir E(S_dir, sigma_dir) (the model applies E to the average)
%   per truth voxel, for f and D: bias of the posterior mean and median, MC SE, RMSE, 90% coverage
%     (central 5-95% interval; nominal 0.90, MC SE ~ sqrt(0.09/Nreal)); posterior mean of noise vs the
%     empirical SD of the averaged datum at the lowest b
%
% Sizes: full Nreal = 200, iteration 20000, burn-in 10000, thinning 10. MCMC_BAYES_T9A_PILOT = 1 runs
% a pilot (Nreal = 10, iteration 2000, burn-in 1000, thinning 5) that only checks the script and
% gives the time per run.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T9a_2_ricianmean_sphericalmean
%
% Kwok-Shing Chan @ MGH
% Date created: 29 September 2026
%

clearvars; tStart = tic;

isPilot     = strcmp(getenv('MCMC_BAYES_T9A_PILOT'), '1');
SNRs        = [5 10];
b           = [1 2.5 5 8];
Ndir        = 30;
[fT, DT]    = ndgrid([0.4 0.6 0.8], [1.8 2.4]);
fT = fT(:).'; DT = DT(:).';
seedData    = [401 402];        % per SNR, shared by all likelihoods
seedRun     = 403;
if isPilot
    Nreal = 10;  iteration = 2000;  burnin = 1000;  thinning = 5;
else
    Nreal = 200; iteration = 20000; burnin = 10000; thinning = 10;
end
NvT     = numel(fT);
Nv      = NvT*Nreal;
Nb      = numel(b);
obj     = gpumcmicro(b);
fwd     = @(p) obj.FWD(p, [], 'mcmc');
variants = {struct('name', 'gaussian',                        'lik', 'gaussian',            'nav', [],   'known', false), ...
            struct('name', 'gaussian_ricianmean (ricianNav)', 'lik', 'gaussian_ricianmean', 'nav', Ndir, 'known', false), ...
            struct('name', 'gaussian_ricianmean (ricianSigma)','lik', 'gaussian_ricianmean', 'nav', [],   'known', true)};

fprintf('T9a.2 Rician-mean Gaussian vs Gaussian likelihood, spherical-mean phantom (characterisation, no pass/fail)%s\n', repmat(' [PILOT]', 1, isPilot));
fprintf('gpumcmicro, b = %s ms/um^2, %d directions per shell, SNR %s (per direction, S0 = 1), %d realisations x %d truth voxels\n', ...
    mat2str(b), Ndir, mat2str(SNRs), Nreal, NvT);
fprintf('iteration %d, burn-in %d, thinning %d; seeds: data %s (per SNR), run %d\n\n', iteration, burnin, thinning, mat2str(seedData), seedRun);

g0 = golden_spiral(Ndir);                           % [3, Ndir] unit vectors
for ks = 1:numel(SNRs)
    SNR = SNRs(ks); sigma = 1/SNR;
    rng(seedData(ks));
    truth.f = repmat(fT, 1, Nreal); truth.D = repmat(DT, 1, Nreal);     % truth index fastest
    n   = randn(3, Nv); n = n ./ sqrt(sum(n.^2, 1));                    % fibre directions
    smClass = double(gather(obj.FWD(truth))).';                         % class spherical mean [Nv, Nb]
    y   = zeros(Nv, Nb); sm = zeros(Nv, Nb); Etrue = zeros(Nv, Nb); Esm = zeros(Nv, Nb);
    for kb = 1:Nb
        for v = 1:Nv
            g   = random_rotation() * g0;
            c2  = (n(:,v).' * g).^2;                                    % [1, Ndir]
            Dr  = (1 - truth.f(v)) * truth.D(v);
            S   = truth.f(v) * exp(-b(kb)*truth.D(v)*c2) + (1 - truth.f(v)) * exp(-b(kb)*(Dr + (truth.D(v) - Dr)*c2));
            S   = S * (smClass(v,kb) / mean(S));                        % exact spherical mean
            M   = abs(S + sigma*randn(1, Ndir) + 1i*sigma*randn(1, Ndir));
            y(v,kb)     = mean(M);
            sm(v,kb)    = mean(S);
            Etrue(v,kb) = mean(mcmc_bayes.rician_mean(S, sigma));
            Esm(v,kb)   = mcmc_bayes.rician_mean(mean(S), sigma);
        end
    end
    fprintf('=== SNR %d (sigma_dir = %.3f, SD of an average ~ %.4f): max |sim - class spherical mean| %.1e ===\n', ...
        SNR, sigma, sigma/sqrt(Ndir), max(abs(sm(:) - smClass(:))));
    fprintf('b = %g: mean data - noise-free spherical mean %+.4f (floor); true expected datum - spherical mean %+.4f; E(SM) - mean E(S_dir) %+.1e\n', ...
        b(end), mean(y(:,end) - sm(:,end)), mean(Etrue(:,end) - sm(:,end)), mean(Esm(:,end) - Etrue(:,end)));
    fprintf('empirical SD of the averaged datum (data - expected) per shell: %s\n\n', mat2str(std(y - Etrue, 0, 1), 3));

    for kv = 1:numel(variants)
        V = variants{kv};
        f = struct();
        f.modelParams = {'f';'D';'noise'};
        f.lb = [1e-6; 1e-6; 1e-3]; f.ub = [1; 3; 0.5]; f.xStepSize = [0.02; 0.05; 0.005];
        f.algorithm = 'MH'; f.likelihood = V.lik;
        if ~isempty(V.nav); f.ricianNav = V.nav; end
        if V.known; f.ricianSigma = sigma; end
        f.parameterTransform = {'sigmoid','sigmoid','log'};
        f.updateScheme = 'joint'; f.adaptStepSize = true; f.adaptInterval = 50;
        f.iteration = iteration; f.burnin = burnin; f.thinning = thinning;
        f.metric = {'mean'};
        x0 = struct('f', 0.5*ones(Nv,1), 'D', 2*ones(Nv,1), 'noise', 0.05*ones(Nv,1));
        [post, out, tRun] = run_chains(y, f, fwd, x0, 1, seedRun);

        % model prediction of the expected datum at the highest b, at the posterior means
        pm  = struct('f', mean(double(post.f), 2).', 'D', mean(double(post.D), 2).');
        nuH = double(gather(obj.FWD(pm))).';                            % [Nv, Nb]
        sH  = mean(double(post.noise), 2);
        switch V.lik
            case 'gaussian'
                yhat = nuH(:,end);
            otherwise
                if V.known; sS = sigma*ones(Nv,1); else; sS = sH*sqrt(Ndir); end
                yhat = mcmc_bayes.rician_mean(nuH(:,end), sS);
        end
        fprintf('--- %s: run %.1f s (%.2f ms/iteration), median acceptance %.3f ---\n', V.name, tRun, 1e3*tRun/iteration, ...
            median(out.diagnostics.acceptance(:)));
        fprintf('b = %g: prediction of the expected datum - truth %+.4f (SD %.4f); posterior mean noise %.4f (empirical SD at b = %g: %.4f)\n', ...
            b(end), mean(yhat - Etrue(:,end)), std(yhat - Etrue(:,end)), mean(sH), b(1), std(y(:,1) - Etrue(:,1)));
        fprintf('%-5s %5s %5s | %9s %7s %8s | %9s | %8s | %6s\n', 'param', 'f', 'D', 'bias', 'rel%', 'se', 'bias(med)', 'RMSE', 'cov90');
        for p = {'f', 'D'}
            xs  = double(post.(p{1}));
            pmv = mean(xs, 2);
            xsr = sort(xs, 2); Ns = size(xs, 2);
            pmd = xsr(:, max(1, round(0.5*Ns)));
            lo  = xsr(:, max(1, round(0.05*Ns))); hi = xsr(:, min(Ns, round(0.95*Ns)));
            for v = 1:NvT
                idx = v:NvT:Nv;
                tr  = truth.(p{1})(v);
                bm  = mean(pmv(idx) - tr); se = std(pmv(idx))/sqrt(Nreal); bmd = mean(pmd(idx) - tr);
                rm  = sqrt(mean((pmv(idx) - tr).^2));
                cv  = mean(lo(idx) <= tr & hi(idx) >= tr);
                fprintf('%-5s %5.2f %5.2f | %+9.4f %+7.1f %8.4f | %+9.4f | %8.4f | %6.3f\n', p{1}, fT(v), DT(v), bm, 100*bm/tr, se, bmd, rm, cv);
            end
        end
        fprintf('\n');
    end
end
fprintf('T9a.2 done (characterisation only, no pass/fail). Total time %.1f s\n', toc(tStart));

% golden-spiral (Fibonacci) set of N unit vectors on the sphere, [3, N]
function g = golden_spiral(N)
k   = (0:N-1) + 0.5;
z   = 1 - 2*k/N;
phi = pi*(1 + sqrt(5))*k;
r   = sqrt(1 - z.^2);
g   = [r.*cos(phi); r.*sin(phi); z];
end

% uniformly random 3D rotation (QR of a Gaussian matrix, sign-corrected, det = +1)
function R = random_rotation()
[Q, Rq] = qr(randn(3));
Q = Q * diag(sign(diag(Rq)));
if det(Q) < 0; Q(:,1) = -Q(:,1); end
R = Q;
end
