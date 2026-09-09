classdef ClassAvailabilityTest < matlab.unittest.TestCase
    % ClassAvailabilityTest - regression test for "is everything actually
    % on the path".
    %
    % This exists specifically because of a real incident: two files
    % required by already-committed code (NEXI/NEXI.m, NEXI/NEXIrotinv.m,
    % MCRMWI/despot1.m) were sitting untracked in the working tree, and
    % nothing caught that a fresh checkout of the repo was broken except
    % manually noticing. This test is deliberately cheap - no GPU, no
    % fitting, just "does this resolve on the path after
    % addpath_gacelle()" - so it can run on every push/PR, including on
    % CI runners with no GPU.
    %
    % Kwok-Shing Chan @ MGH

    properties (TestParameter)
        % Every supported (non-sandbox, non-deprecated) model class.
        modelClass = struct( ...
            'R2starMapping',        'gpuR2starMapping', ...
            'NEXI',                 'gpuNEXI', ...
            'AxCaliberSMT',         'gpuAxCaliberSMT', ...
            'GREMWI',                'gpuGREMWI', ...
            'MCRMWI',                'gpuMCRMWI', ...
            'JointR1R2starMapping', 'gpuJointR1R2starMapping', ...
            'SANDI',                 'gpuSANDI', ...
            'mcmicro',                'gpumcmicro', ...
            'mcTFI',                 'gpumcTFI', ...
            'PDF',                    'gpuPDF' ...
            )
        % The two shared solvers every model class dispatches to.
        solverClass = struct('askadam', 'askadam', 'mcmc', 'mcmc')
    end

    methods (TestClassSetup)
        function addGacellePath(testCase)
            testsDir    = fileparts(mfilename('fullpath'));
            projectRoot = fileparts(testsDir);
            % addpath_gacelle.m itself lives at the project root, so it
            % must be reachable before we can call it.
            addpath(projectRoot);
            testCase.assertEqual(exist('addpath_gacelle', 'file'), 2, ...
                'addpath_gacelle.m should be found at the project root.');
            addpath_gacelle(projectRoot);
        end
    end

    methods (Test)
        function testModelClassResolves(testCase, modelClass)
            testCase.verifyEqual(exist(modelClass, 'class'), 8, ...
                sprintf(['%s should resolve to a class on the MATLAB path ' ...
                'after addpath_gacelle() - if this fails, a required .m ' ...
                'file is likely missing or untracked.'], modelClass));
        end

        function testSolverResolves(testCase, solverClass)
            testCase.verifyEqual(exist(solverClass, 'class'), 8, ...
                sprintf('%s should resolve to a class on the MATLAB path after addpath_gacelle().', solverClass));
        end

        function testAddpathGacelleExcludesNonReleasePaths(testCase)
            % Regression-proof addpath_gacelle's own exclusion logic:
            % docs/, sandbox/, deprecated/ and mpl_training/ folders
            % should never end up on the path.
            testsDir    = fileparts(mfilename('fullpath'));
            projectRoot = fileparts(testsDir);
            onPath      = strsplit(path, pathsep);
            gacellePaths = onPath(startsWith(onPath, projectRoot));

            testCase.verifyTrue(~any(contains(gacellePaths, [filesep 'docs'])), ...
                'No path under docs/ should be added by addpath_gacelle().');
            testCase.verifyTrue(~any(contains(gacellePaths, 'sandbox')), ...
                'No path containing "sandbox" should be added by addpath_gacelle().');
            testCase.verifyTrue(~any(contains(gacellePaths, 'deprecated')), ...
                'No path containing "deprecated" should be added by addpath_gacelle().');
            testCase.verifyTrue(~any(contains(gacellePaths, 'mpl_training')), ...
                'No path containing "mpl_training" should be added by addpath_gacelle().');
        end

        function testGACELLEVersionCallable(testCase)
            % Regression-proof the docs' claim that utils.GACELLE_version()
            % is a real, callable helper (it used to be a same-named but
            % unrelated standalone script, not a method of utils).
            v = utils.GACELLE_version();
            testCase.verifyClass(v, 'char');
            testCase.verifyNotEmpty(v);
        end
    end
end
