%% run_tests.m
%
% Convenience runner for GACELLE's regression test suite.
%
%   cd tests; run_tests
%
% Adds GACELLE (and the tests/ folder itself) to the path, runs every
% test in tests/, and prints a summary. Tier-2 (GPU-dependent) tests
% report as "Incomplete"/filtered rather than failed on machines with no
% GPU - that's expected, not a problem.
%
% Kwok-Shing Chan @ MGH

testsDir    = fileparts(mfilename('fullpath'));
projectRoot = fileparts(testsDir);

addpath(projectRoot);
addpath_gacelle(projectRoot);
addpath(testsDir);

results = runtests(testsDir, 'IncludeSubfolders', true);

disp(table(results));

nFailed  = nnz([results.Failed]);
nSkipped = nnz([results.Incomplete]) - nFailed; % filtered tests report Incomplete too
nPassed  = numel(results) - nFailed - nSkipped;

fprintf('\n%d passed, %d failed, %d skipped (out of %d)\n', ...
    nPassed, nFailed, nSkipped, numel(results));

if nFailed > 0
    error('run_tests:failures', '%d test(s) failed.', nFailed);
end
