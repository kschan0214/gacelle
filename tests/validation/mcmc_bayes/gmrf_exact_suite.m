function [isPass, labels, res] = gmrf_exact_suite(update, iteration)
% GMRF_EXACT_SUITE Run all configurations of the exact Gaussian-MRF test (T4.2) with the given
% MRF update ('chromatic', or 'simultaneous' = TEST ONLY negative control of T4.3), print the
% per-configuration results and return the T4.2 criterion per configuration.
%
%   [isPass, labels, res] = gmrf_exact_suite(update, iteration)
%
% Criterion (see test_T4_2_gaussian_mrf_exact.m), per configuration:
%   frac(|z|>1.96) <= 0.10, max|z| <= z*(N) = sqrt(2) erfcinv(0.01/N), median|z_noMRF| >= 3,
%   scaling check <= 1e-10. Burn-in 20% of the iterations, thinning 5, 4 chains.
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%
burnin = round(0.2*iteration); thinning = 5;

%            mode  r  conn    tau   scheme           label
cfgs = { {'3d', 1, 'face',   5,   'joint',         '3d face, tau 5 (moderate)'}, ...
         {'3d', 1, 'face',   0.5, 'joint',         '3d face, tau 0.5 (strong)'}, ...
         {'3d', 1, 'face',   0.5, 'componentwise', '3d face, tau 0.5 (strong), componentwise'}, ...
         {'3d', 1, 'face',   0.1, 'joint',         '3d face, tau 0.1 (very strong)'}, ...
         {'2d', 2, 'full',  15,   'joint',         '2d r=2, tau 15 (moderate)'}, ...
         {'2d', 2, 'full',   1.5, 'joint',         '2d r=2, tau 1.5 (strong)'}, ...
         {'2d', 2, 'full',   0.3, 'joint',         '2d r=2, tau 0.3 (very strong)'} };
seeds = 101:100+numel(cfgs);

fprintf('T4.2 exact Gaussian MRF (update: %s): %d iterations, burn-in %d, thinning %d, 4 chains\n', update, iteration, burnin, thinning);
fprintf('Seeds: geometry/design/data 91/92/93, scaling check 94, runs %s (rng and gpu rng)\n', mat2str(seeds));
fprintf('Criterion per configuration: frac(|z|>1.96) <= 0.10, max|z| <= z*(N) (Bonferroni 1%%), median|z_noMRF| >= 3, scaling check <= 1e-10\n\n');

isPass = false(1, numel(cfgs)); res = cell(1, numel(cfgs));
for kc = 1:numel(cfgs)
    c = cfgs{kc};
    cfg = struct('mode', c{1}, 'radius', c{2}, 'connectivity', c{3}, 'tau', c{4}, 'updateScheme', c{5}, ...
                 'mrfUpdate', update, 'iteration', iteration, 'burnin', burnin, 'thinning', thinning, ...
                 'Nchain', 4, 'seedRun', seeds(kc));
    R = gmrf_exact_run(cfg);
    res{kc} = R;
    zStar = sqrt(2)*erfcinv(0.01/R.N);
    frac  = mean(abs(R.z) > 1.96); zmax = max(abs(R.z));
    isPass(kc) = frac <= 0.10 && zmax <= zStar && R.zFlatMedian >= 3 && R.scalingCheck <= 1e-10;
    fprintf('[%d] %s: Nv %d, edges %d, colours %d, run %.0f s\n', kc, c{6}, R.Nv, R.Nedges, R.settings.NcoloursUsed, R.tRun);
    fprintf('    mean diagonal precision: local %.2f, MRF %.2f | exact neighbour correlation (median) %.3f | scaling check %.1e\n', ...
        R.priorVsMrfDiag(1), R.priorVsMrfDiag(2), R.nbrCorrExact, R.scalingCheck);
    for q = 1:numel(R.cat)
        zq = R.cat(q).z;
        fprintf('    %-17s n=%5d  frac|z|>1.96 %.3f  max|z| %5.2f  median z %+.2f\n', R.cat(q).name, numel(zq), mean(abs(zq)>1.96), max(abs(zq)), median(zq));
    end
    fprintf('    pooled N=%d: frac %.3f, max|z| %.2f (z* %.2f) | median|z| vs no-MRF means %.1f | median|z| nbr cov vs 0 %.1f\n', ...
        R.N, frac, zmax, zStar, R.zFlatMedian, R.zN0Median);
    fprintf('    sampler/exact: variance ratio median %.3f [%.3f, %.3f], neighbour cov ratio median %.3f [%.3f, %.3f]\n', ...
        R.varRatio, R.nbrCovRatio);
    fprintf('    R-hat max %.4f | ESS median %.0f, min %.0f | acceptance median %s\n', R.rhatMax, R.essMedian, R.essMin, mat2str(R.accMedian, 3));
    fprintf('    -> %s\n\n', pf(isPass(kc)));
end
labels = cellfun(@(c) c{6}, cfgs, 'UniformOutput', false);
end

function s = pf(c)
if c; s = 'PASS'; else; s = 'FAIL'; end
end
