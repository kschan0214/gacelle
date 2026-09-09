classdef SmokeFit_gpuPDFTest < matlab.unittest.TestCase
    % Tier-2 smoke test for gpuPDF: fit a tiny synthetic background-field
    % removal problem and check the fit runs and produces finite output.
    %
    % Unlike every other model in this suite, gpuPDF has no
    % demo_*NoisePropagation*.m to base a test on, so this instead follows
    % the design SEPIA's own test suite uses for its background-field-
    % removal tests (sepia/test/phantom/generate_synthetic_phantom.m): a
    % couple of point susceptibility sources OUTSIDE a small ellipsoid
    % "brain" mask, dipole-kernel-convolved (via gpuPDF's own static
    % dipole_kernel method) to a background field in Hz, plus noise. gapMinMM
    % is shrunk from its 20 mm default so the internal zero-padding stays
    % small and the test stays fast.
    %
    % Kwok-Shing Chan @ MGH

    methods (TestClassSetup)
        function addGacellePath(testCase)
            testsDir    = fileparts(mfilename('fullpath'));
            projectRoot = fileparts(testsDir);
            addpath(projectRoot);
            addpath_gacelle(projectRoot);
        end
    end

    methods (Test)
        function testAskadamFitRunsAndIsFinite(testCase)
            gacelletest.assumeGPU(testCase);

            seed = 1; rng(seed);

            matrixSize = [10 10 8];
            voxelSize  = [2 2 2]; % mm
            B0    = 3;            % T
            B0dir = [0 0 1];

            [x, y, z] = ndgrid(1:matrixSize(1), 1:matrixSize(2), 1:matrixSize(3));
            cx = (matrixSize(1)+1)/2; cy = (matrixSize(2)+1)/2; cz = (matrixSize(3)+1)/2;

            % synthetic "brain" ROI mask
            mask = ((x-cx)/(matrixSize(1)*0.35)).^2 + ((y-cy)/(matrixSize(2)*0.35)).^2 ...
                 + ((z-cz)/(matrixSize(3)*0.35)).^2 <= 1;

            % a couple of susceptibility sources OUTSIDE the mask (the only
            % place gpuPDF's fitted parameter is supported)
            chi_true = zeros(matrixSize);
            chi_true(2, 2, 2)             = 0.08;
            chi_true(end-1, end-1, end-1) = -0.05;
            chi_true = chi_true .* ~mask;

            D = gpuPDF.dipole_kernel(matrixSize, voxelSize, B0dir);
            backgroundField_ppm = real(ifftn(D .* fftn(chi_true)));
            backgroundField_Hz  = backgroundField_ppm * gpuPDF.gyro * B0;

            noise = 0.5 * randn(matrixSize); % Hz
            totalField_Hz = backgroundField_Hz + noise;

            objGPU = gpuPDF(voxelSize, B0, B0dir);
            objGPU.gapMinMM = 4; % keep the internal zero-padding small for speed

            fitting              = [];
            fitting.solver       = 'askadam';
            fitting.iteration    = 30;
            extraData            = [];

            out = objGPU.estimate(totalField_Hz, mask, extraData, fitting);

            testCase.verifyTrue(isfield(out, 'final'));
            for name = {'chi_b', 'backgroundField', 'localField'}
                testCase.verifyTrue(isfield(out.final, name{1}));
                testCase.verifyTrue(all(isfinite(out.final.(name{1})(:))));
            end
        end
    end
end
