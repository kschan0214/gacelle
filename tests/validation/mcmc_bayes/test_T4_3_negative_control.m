%% test_T4_3_negative_control.m
%
% T4.3 (Phase 4): negative control of T4.2. The same configurations, data, seeds and criterion
% as test_T4_2_gaussian_mrf_exact.m (gmrf_exact_suite.m), but with the TEST ONLY update
% fitting.mrfUpdate = 'simultaneous': all voxels are proposed and accepted at once, each with
% its neighbours taken from the pre-sweep state. This is not a valid MH kernel for the joint
% target (it is the reference implementation's scheme, with fixed weights), so a sensitive
% T4.2 must reject it.
%
% Criterion (stated before running): T4.3 PASS if the T4.2 criterion FAILS in every "strong"
%   and "very strong" configuration (3d tau 0.5 joint and componentwise, 3d tau 0.1, 2d tau 1.5,
%   2d tau 0.3). The "moderate" configurations are reported only (the error of the simultaneous
%   update shrinks with the coupling, so the test may be insensitive there). If a strong
%   configuration passes, T4.2 is reported as insensitive for it (T4.2 is not weakened).
%   How badly it fails is shown by the z summaries and the sampler/exact ratios of the marginal
%   variances and neighbour covariances.
%
% Run from the repository root with MATLAB R2024b (GPU):
%   addpath(pwd); addpath_gacelle(pwd);
%   addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
%   test_T4_3_negative_control
% MCMC_BAYES_T4_ITER overrides the number of iterations (pilot runs).
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%

clearvars; tStart = tic;

iteration = 30000;
if ~isempty(getenv('MCMC_BAYES_T4_ITER')); iteration = str2double(getenv('MCMC_BAYES_T4_ITER')); end
[t42Pass, labels] = gmrf_exact_suite('simultaneous', iteration);

isStrong = contains(labels, 'strong');
fprintf('T4.3 summary (T4.2 criterion under the simultaneous update; must FAIL where strong):\n');
for k = 1:numel(labels)
    if isStrong(k); tag = 'required to fail'; else; tag = 'information'; end
    fprintf('  %-45s T4.2 criterion %s  (%s)\n', labels{k}, pf(t42Pass(k)), tag);
end
isPass = ~any(t42Pass(isStrong));
if ~isPass
    fprintf('  T4.2 is INSENSITIVE to the simultaneous update in: %s\n', strjoin(labels(isStrong & t42Pass), '; '));
end
fprintf('T4.3 overall: %s   (total time %.1f min)\n', pf(isPass), toc(tStart)/60);

function s = pf(c)
if c; s = 'PASS'; else; s = 'FAIL'; end
end
