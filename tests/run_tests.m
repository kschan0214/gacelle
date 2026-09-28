%% run_tests.m
%
% Convenience runner for GACELLE's regression test suite.
%
%   cd tests; run_tests
%
% Adds GACELLE (and the tests/ folder itself) to the path, runs every
% test in tests/ (excluding tests/validation/), and prints a summary. Tier-2 (GPU-dependent) tests
% report as "Incomplete"/filtered rather than failed on machines with no
% GPU - that's expected, not a problem.
%
% Kwok-Shing Chan @ MGH

testsDir    = fileparts(mfilename('fullpath'));
projectRoot = fileparts(testsDir);

addpath(projectRoot);
addpath_gacelle(projectRoot);
addpath(testsDir);

% build the suite from tests/ and its subfolders, but exclude anything
% under tests/validation/ (long, hand-run statistical scripts, see
% tests/validation/*/README.md)
suite   = matlab.unittest.TestSuite.fromFolder(testsDir, 'IncludingSubfolders', true);
isValidation = contains({suite.BaseFolder}, [filesep 'validation']);
suite   = suite(~isValidation);

results = run(suite);

disp(table(results));

nFailed  = nnz([results.Failed]);
nSkipped = nnz([results.Incomplete]) - nFailed; % filtered tests report Incomplete too
nPassed  = numel(results) - nFailed - nSkipped;

fprintf('\n%d passed, %d failed, %d skipped (out of %d)\n', ...
    nPassed, nFailed, nSkipped, numel(results));

if nFailed > 0
    error('run_tests:failures', '%d test(s) failed.', nFailed);
end
