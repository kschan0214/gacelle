classdef SmokeFit_gpumcTFITest < matlab.unittest.TestCase
    % Tier-2 smoke test for gpumcTFI: fit a tiny synthetic complex
    % multi-echo GRE dataset and check the joint (M0, R2*, susceptibility,
    % phase offset) fit runs and produces finite output.
    %
    % Like gpuPDF, gpumcTFI has no demo_*NoisePropagation*.m to base a
    % test on, so this follows the same SEPIA-inspired synthetic-phantom
    % design used in SmokeFit_gpuPDFTest.m (see
    % sepia/test/phantom/generate_synthetic_phantom.m): a couple of point
    % susceptibility sources dipole-kernel-convolved to a local field,
    % combined with a mono-exponential R2* decay to build the multi-echo
    % complex signal directly (mcTFI takes raw complex GRE, not a
    % pre-processed field map - unlike gpuPDF, estimate() derives R2*, M0
    % and the dipole kernel from the data itself, so extraData can stay
    % empty). gapMinMM is shrunk from its 20 mm default so the internal
    % zero-padding stays small and the test stays fast.
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

            matrixSize = [8 8 6];
            voxelSize  = [2 2 2]; % mm
            B0    = 3;            % T
            B0dir = [0 0 1];
            Nt    = 4;
            te    = linspace(2e-3, 20e-3, Nt); % s

            [x, y, z] = ndgrid(1:matrixSize(1), 1:matrixSize(2), 1:matrixSize(3));
            cx = (matrixSize(1)+1)/2; cy = (matrixSize(2)+1)/2; cz = (matrixSize(3)+1)/2;

            % synthetic "head" mask (fidelity region; susceptibility itself
            % is estimated over the whole FOV, not just here)
            mask = ((x-cx)/(matrixSize(1)*0.4)).^2 + ((y-cy)/(matrixSize(2)*0.4)).^2 ...
                 + ((z-cz)/(matrixSize(3)*0.4)).^2 <= 1;

            % ground-truth susceptibility sources -> local field, in ppm
            chi_true = zeros(matrixSize);
            chi_true(3, 3, 3)             = 0.06;
            chi_true(end-2, end-2, end-2) = -0.04;

            D = gpuPDF.dipole_kernel(matrixSize, voxelSize, B0dir);
            f_ppm = real(ifftn(D .* fftn(chi_true)));

            % mono-exponential magnitude decay + susceptibility-driven phase
            S0        = 1 + 0.2 * rand(matrixSize);
            R2starGT  = 20 + 10 * rand(matrixSize); % 1/s
            gyro      = gpuPDF.gyro;

            data = zeros([matrixSize Nt]);
            for e = 1:Nt
                magn  = S0 .* exp(-R2starGT * te(e));
                phase = 2 * pi * gyro * B0 .* f_ppm .* te(e);
                noise = 0.02 * (randn(matrixSize) + 1i * randn(matrixSize));
                data(:, :, :, e) = magn .* exp(1i * phase) + noise;
            end

            objGPU = gpumcTFI(voxelSize, B0, B0dir, te);
            objGPU.gapMinMM = 4; % keep the internal zero-padding small for speed

            fitting              = [];
            fitting.solver       = 'askadam';
            fitting.iteration    = 30;
            extraData            = [];

            out = objGPU.estimate(data, mask, extraData, fitting);

            testCase.verifyTrue(isfield(out, 'final'));
            for name = {'M0', 'R2star', 'chi', 'fittedField'}
                testCase.verifyTrue(isfield(out.final, name{1}));
                testCase.verifyTrue(all(isfinite(out.final.(name{1})(:))));
            end
        end
    end
end
