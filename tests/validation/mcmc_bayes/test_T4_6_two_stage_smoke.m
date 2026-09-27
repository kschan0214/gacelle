%% test_T4_6_two_stage_smoke.m
%
% T4.6 (Phase 4, optional): end-to-end smoke run of mcmc_bayes.run_two_stage on the small IVIM
% phantom (ivim_mrf_phantom.m, SNR 30), 'marginal_S0noise', log D, sigmoid F, log Dstar.
% NOT a benchmark: it only checks that both stages run and give sane output.
%   Stage 1: free NIW hierarchical prior (default hyperprior), on 50% of the voxels.
%   Stage 2: mu, Sigma fixed at the stage-1 posterior means, + L1 MRF ('3d' face, tau = 1).
%   One chain, 8000 iterations, burn-in 2000, thinning 4.
% Sanity criteria (stated before running): all posterior means finite and inside the box;
%   settings.empiricalBayes and settings.mrf present; stage-2 Sigma equals the stage-1 posterior
%   mean; a loose physical check of stage-1 mu: exp(mu) of D in [0.5, 1.5] um^2/ms and of Dstar
%   in [8, 40] um^2/ms; the RMSE of the stage-2 posterior
%   mean of D is not larger than 1.5 x the RMSE of a hierarchical-only (fixed, same mu/Sigma) run.
%   Printed: hyperparameters, RMSE per parameter (two-stage vs hierarchical only), runtimes.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T4_6_two_stage_smoke
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%

clearvars; tStart = tic;
seedPhantom = 301; seedRun = 401;
[y, mask, truth, f] = ivim_mrf_phantom(30, seedPhantom);
b = f.b; f = rmfield(f, 'b');
f.iteration = 8000; f.burnin = 2000; f.thinning = 4;
f.prior.hierarchical = struct('subsetFraction', 0.5);
f.prior.mrf = struct('potential', 'l1', 'tau', 1, 'mode', '3d');
x0 = struct('S0', 1, 'D', 1, 'F', 0.1, 'Dstar', 20, 'noise', 0.03);
fwd = @(pp) ivim_fwd(pp, b);
idx = find(mask);

rng(seedRun); parallel.gpu.rng(seedRun);
t0 = tic;
evalc('out = mcmc_bayes().run_two_stage(y, mask, [], x0, f, fwd);');
tTwo = toc(t0);
eb = out.settings.empiricalBayes;

% hierarchical only with the same fixed mu/Sigma (reference for the RMSE check)
g = f; g.prior = struct('hierarchical', struct('fixed', true, 'mu', eb.muHat, 'Sigma', eb.SigmaHat));
rng(seedRun); parallel.gpu.rng(seedRun);
t0 = tic;
evalc('outH = mcmc_bayes().optimisation(y, mask, [], x0, g, fwd);');
tH = toc(t0);

names = {'D','F','Dstar'}; lb = [0 0 0]; ub = [Inf 1 Inf];
ok = isfield(out.settings,'mrf') && ~isempty(out.settings.mrf) && isequal(out.hyper.mean.Sigma, eb.SigmaHat) ...
     && isequal(eb.SigmaHat, out.stage1.hyper.mean.Sigma);
rm = zeros(2, 3);
for p = 1:3
    m2 = out.mean.(names{p})(idx); mH = outH.mean.(names{p})(idx); tv = truth.(names{p})(idx);
    ok = ok && all(isfinite(m2)) && all(m2 > lb(p) & m2 < ub(p));
    rm(:,p) = [sqrt(mean((m2 - tv).^2)); sqrt(mean((mH - tv).^2))];
end
muN = [exp(eb.muHat(1)), 1/(1+exp(-eb.muHat(2))), exp(eb.muHat(3))];
ok = ok && muN(1) > 0.5 && muN(1) < 1.5 && muN(3) > 8 && muN(3) < 40 && rm(1,1) <= 1.5*rm(2,1);

fprintf('T4.6 two-stage smoke (IVIM phantom, Nv %d, stage 1 on %d voxels)\n', numel(idx), eb.Nstage1);
fprintf('stage-1 mu (native: D %.3f, F %.3f, Dstar %.2f); SigmaHat diag %s\n', muN, mat2str(diag(eb.SigmaHat).', 3));
fprintf('stage-2 MRF: %s tau %g, W %s, %d colours, %d edges, subsetForward used %d\n', out.settings.mrf.potential, out.settings.mrf.tau, ...
    mat2str(out.settings.mrf.W.', 3), out.settings.mrf.NcoloursUsed, out.settings.mrf.Nedges, out.settings.mrf.subsetForward.used);
fprintf('RMSE of the posterior mean   D       F       Dstar\n');
fprintf('  two-stage (hier + MRF)   %.4f  %.4f  %.3f\n', rm(1,:));
fprintf('  hierarchical only        %.4f  %.4f  %.3f\n', rm(2,:));
fprintf('median acceptance stage 2 %.3f; runtime two-stage %.0f s, hierarchical only %.0f s\n', ...
    median(out.diagnostics.acceptance(mask)), tTwo, tH);
fprintf('T4.6 smoke: %s   (total time %.1f min)\n', pf(ok), toc(tStart)/60);

function s = pf(c)
if c; s = 'PASS'; else; s = 'FAIL'; end
end
