classdef gpuPDF < handle
% Projection onto Dipole Fields (PDF) background field removal.
%
% Estimates a background susceptibility distribution chi_b supported OUTSIDE
% the ROI mask, such that its dipole field best explains the measured total
% field INSIDE the mask. The local field is then obtained by subtraction.
%
% Two data fidelity formulations are available via fitting.isnonlinear:
%
%   false (default) : linear PDF. Residual is taken on the field itself,
%                     in ppm. Convex quadratic; equivalent to the branch
%                     that FANSI's npdfCG.m actually executes.
%
%   true            : nonlinear PDF. Residual is taken between unit phasors,
%                     exp(i*phi) vs exp(i*phi_measured), implemented as a
%                     stacked [cos, sin] pair so that the built-in L2 loss
%                     reproduces |exp(i*a)-exp(i*b)|^2 = 2(1-cos(a-b)).
%                     This makes the fidelity 2*pi-periodic (tolerant to
%                     residual phase wraps) and bounded per voxel (outliers
%                     cannot dominate the gradient). Requires dTE.
%
% NOTE ON SUPPORT: this model inverts the usual GACELLE convention. The
% fitted parameter lives on (1-mask), while the loss is evaluated on mask.
% Parameter support and loss support are therefore complementary, which is
% why isOptimiseMemory must remain false - masked parameter storage assumes
% the two coincide.
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
%
% Support askadam.m only!
%
% Date created: 18 July 2026
% Date modified:

    properties (GetAccess = public, SetAccess = protected)
    % ===== MODEL PARAMETER CONTRACT =====
    % chi_b     : background susceptibility [ppm]
    %
    % modelParams{k} <-> ub(k) <-> lb(k) <-> startPoint(k) <-> step(k)
    % These five arrays MUST stay the same length and index-aligned.
    % Mutate only as a set, via updateProperty() - never assign into a
    % single element from outside the class, or these will desync.
    %
    % Unlike the other models in the framework there is no solver-conditional
    % 'noise' parameter here, because mcmc.m is not supported (see
    % check_set_default). Any future solver-conditional parameter should be
    % appended LAST so that updateProperty() can strip it by name.
    %
    % Bounds are in ppm and must accommodate air (~9.4 ppm relative to
    % tissue), which is the dominant background source.
        modelParams     = { 'chi_b';};
        ub              = [  10;    ];
        lb              = [ -10;    ];
        startPoint      = [   0;    ];
        step            = [ 0.1;    ];
    end

    properties
    % ===== USER-TUNABLE OPTIONS =====
    % Freely settable by users before fitting; no coupling between these.

        gapMinMM    = 20; % mm

        seed = 48463;   % for reproducible random number generation

        epsilon = utils.epsilon;

    end

    properties (GetAccess = public, SetAccess = protected)
    % ===== ACQUISITION PARAMETERS =====
    % Set once in the constructor from user-provided acquisition info.
    % Read-only after construction.

        voxelSize;      % [mm], 1x3
        B0dir;          % unit vector [x,y,z] of B0 in image coordinates
        B0;             % [T]
        dTE;            % [s], echo spacing / effective TE. Only used by the
                        % nonlinear branch, to convert ppm -> rad.
        phi2;           % [rad per ppm] = 2*pi*gyro*B0*dTE. Derived in the
                        % constructor; do not set directly.
    end

    properties (Constant)
            gyro = 42.57747892;     % [MHz/T]
    end

    methods

        % constructor
        function this = gpuPDF(voxelSize,B0,B0dir,dTE)
        % obj = gpuPDF(voxelSize, B0, B0dir, dTE)
        %
        % Input
        % ----------
        % voxelSize : voxel size [mm], 1x3
        % B0        : main field strength [T]
        % B0dir     : static field direction [x,y,z]
        % dTE       : echo spacing / effective TE [s]. Optional for the
        %             linear branch; REQUIRED for fitting.isnonlinear = true,
        %             where it sets the ppm -> rad conversion. Defaults to 0,
        %             which makes phi2 = 0 and renders the nonlinear forward
        %             degenerate (cos->1, sin->0 everywhere).
        %
        % Output
        % ----------
        % this      : object of a fitting class
        %
        % Author:
        %  Kwok-Shing Chan (kchan2@mgh.harvard.edu)
        %  Copyright (c) 2023 Massachusetts General Hospital
        %
            this.voxelSize   = single(voxelSize(:)).';
            this.B0dir       = single(B0dir(:)).';
            this.B0          = single(B0);

            if nargin < 4
                this.dTE = 0;
            else
                this.dTE = single(dTE);
            end

            % rad per ppm: gyro [MHz/T] * B0 [T] = Hz per ppm; * 2*pi*dTE -> rad per ppm
            this.phi2 = 2*pi*this.gyro*this.B0*this.dTE;

        end


        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('============================');
            disp('Projection onto dipole field');
            disp('============================');


            fprintf('\n')

        end

        %% higher-level data fitting functions
        % Wrapper function of fit to handle image data
        function  [out] = estimate(this, data, mask, extraData, fitting)
        % Perform PDF background field removal based on askAdam
        %
        % Input
        % -----------
        % data      : 3D total field map in Hz, [x,y,z]
        % mask      : 3D ROI mask, [x,y,z]. Defines BOTH the region where the
        %             data fidelity is evaluated AND (as its complement) the
        %             support of the fitted background susceptibility. These
        %             two roles are not separable - eroding/dilating the mask
        %             moves both.
        % extraData : Optional additional data
        %   .weights: 3D data weighting matrix (typically magnitude,
        %             normalised to max 1 so that tol is dataset-portable)
        % fitting   : fitting algorithm parameters (see fit function)
        %
        % Output
        % -----------
        % out       : output structure contains all estimation results
        %

            % display basic info
            this.display_data_model_info;

            % get all fitting algorithm parameters
            fitting = this.check_set_default(fitting);

            % build the dipole kernel and cast extraData to single
            [ data, mask, extraData ] = this.prepare_data(data, mask, extraData);

            %%%%%%%%%%%%%%%% Step 2: Memory management %%%%%%%%%%%%%%%%
            % No segmentation here: the dipole convolution is global, so the
            % volume cannot be split along any spatial axis without changing
            % the forward operator.

            % parameter estimation
            [out] = this.fit(data,mask,fitting,extraData);

            %%%%%%%%%%%%%%%% End Step 2 %%%%%%%%%%%%%%%%

            % save the estimation results if the output filename is provided
            askadam.save_askadam_output(fitting.outputFilename,out)

        end

        % Data fitting function, image-based (3D)
        function [out] = fit(this,f_Hz,mask,fitting,extraData)
        %
        % Input
        % -----------
        % f_Hz      : 3D total field map [Hz], [x,y,z]
        % mask      : 3D ROI mask, [x,y,z] (see note in estimate)
        % fitting   : fitting algorithm parameters
        %
        % Output
        % -----------
        % out       : output structure
        %   .final      : final results
        %       .chi_b          : background susceptibility [ppm], on (1-mask)
        %       .backgroundField: dipole field of chi_b [Hz]
        %       .localField     : f_Hz - backgroundField [Hz]
        %       .loss           : final loss metric
        %   .min        : as above, at the iteration with minimum loss
        %
        % Units: chi_b is returned in ppm regardless of branch. The
        % background/local fields are returned in Hz, i.e. the same units as
        % the input, so no conversion is needed by the caller.
        %
        % Kwok-Shing Chan @ MGH
        % kchan2@mgh.harvard.edu
        % Date created: 18 July 2026
        % Date modified:
        %

            % get image size
            dims = size(f_Hz,1:3);

            %%%%%%%%%%%%%%%%%%%% 1. Validate and parse input %%%%%%%%%%%%%%%%%%%%
            if nargin < 3 || isempty(mask); mask = ones(dims,'logical'); end % if no mask input then fit everthing
            if nargin < 4; fitting = struct(); end

            % get all fitting algorithm parameters
            fitting                 = this.check_set_default(fitting);
            % determine fitting parameters
            fitting.modelParams     = this.modelParams;
            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(this.modelParams)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(this.modelParams)); end

            % set initial starting points
            pars0                   = this.determine_x0(f_Hz,fitting);

            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w = this.compute_optimisation_weights(extraData,f_Hz,fitting); % This is a customised funtion

            % 2.2 put the measurement into the same representation as the
            % forward model output, so that residuals are like-for-like.
            %   linear    : forward emits ppm      -> convert Hz to ppm
            %   nonlinear : forward emits [cos,sin] of the phase -> build the
            %               same phasor pair from the measured field. The
            %               radians are computed directly from Hz rather than
            %               via ppm; the two are algebraically identical
            %               (phi2 * f_ppm == 2*pi*dTE * f_Hz) but this keeps
            %               one less factor in play.
            switch fitting.isnonlinear
                case false
                    f   = f_Hz ./ (this.gyro * this.B0);              % ppm
                case true
                    phi = 2*pi*this.dTE .* f_Hz;                      % rad
                    f   = cat(4, cos(phi), sin(phi));                 % dimensionless, [x,y,z,2]
            end

            % 2.3 display optimisation algorithm parameters
            this.display_algorithm_info(fitting)

            % 2.4 askAdam optimisation main
            % NOTE: mask is passed twice - once as the loss mask (2nd arg)
            % and once through to FWD (7th arg), where its complement defines
            % the parameter support.
            switch fitting.solver
                case 'askadam'

                    out         = askadam().optimisation(f, mask, w, pars0, fitting, @this.FWD, mask, fitting,extraData);

                    % post processing
                    out.final   = this.postprocess(out.final, mask,  extraData, f_Hz);
                    out.min     = this.postprocess(out.min, mask,  extraData, f_Hz);

                    % % convert the estimated chi_b [ppm] back to a field in Hz
                    % % and subtract to obtain the local field
                    % backgroundField_Hz          = real(gacelleFFT.ifft3s_(extraData.D .* gacelleFFT.fft3s_(out.final.chi_b))) .* (this.gyro * this.B0);
                    % localField_Hz               = f_Hz - backgroundField_Hz;
                    % out.final.backgroundField   = backgroundField_Hz;
                    % out.final.localField        = localField_Hz;
                    % 
                    % backgroundField_Hz          = real(gacelleFFT.ifft3s_(extraData.D .* gacelleFFT.fft3s_(out.min.chi_b))) .* (this.gyro * this.B0);
                    % localField_Hz               = f_Hz - backgroundField_Hz;
                    % out.min.backgroundField     = backgroundField_Hz;
                    % out.min.localField          = localField_Hz;
                    % 
                    % out.mask = mask;
            end

            disp('The process is completed.')
            disp('##############################################')

            % clear GPU
            reset(gpuDevice)

        end

        % compute weights for optimisation
        function w = compute_optimisation_weights(this,extraData,data,fitting)
        %
        % Output
        % ------
        % w         : ND signal masked weights
        %
        % Typically the magnitude image. Normalise to max 1 before passing in:
        % the weight multiplies the residual before squaring, so an arbitrary
        % magnitude scale shifts the loss by that scale squared and makes tol
        % non-portable across datasets.

            if ~isempty(extraData) && isfield(extraData,'weights')
                w = extraData.weights;
            else
                w = ones(size(data));
            end

            % replicate along the phasor dimension so the weight applies to
            % both the cos and sin channels of the nonlinear residual
            if fitting.isnonlinear
                w = repmat(w,1,1,1,2);
            end
        end

        %% Prior estimation related functions

        % determine how the starting points will be set up
        function x0 = determine_x0(this,y,fitting)
        % Uniform starting point from this.startPoint.
        %
        % For the linear branch this is harmless - the objective is convex.
        % For the nonlinear branch the objective is periodic and non-convex,
        % so a poor starting point can settle in the wrong wrap. A scaled
        % background estimate (cf. FANSI's backmodel/fitmodels initialisation)
        % is preferable there.

            disp('---------------');
            disp('Starting points');
            disp('---------------');

            dims = size(y,1:3);

            % use fixed points
            fprintf('Using default starting points for all voxels at [%s]: [%s]\n', cell2str(this.modelParams),replace(num2str(this.startPoint(:).',' %.2f'),' ',','));
            x0 = utils.initialise_x0(dims,this.modelParams,this.startPoint);


            % make sure the input is bounded
            x0 = askadam.set_boundary(x0,fitting.ub,fitting.lb);

            fprintf('Estimation lower bound [%s]: [%s]\n',      cell2str(this.modelParams),replace(num2str(fitting.lb(:).',' %.2f'),' ',','));
            fprintf('Estimation upper bound [%s]: [%s]\n',      cell2str(this.modelParams),replace(num2str(fitting.ub(:).',' %.2f'),'  ',','));
            ('---------------');
        end

        %% Signal related functions

        % compute the forward model
        function [f] = FWD(this, pars, mask, fitting, extraData)
        % Forward model: background susceptibility -> field.
        %
        % Output units depend on the branch:
        %   linear    : ppm            , [x,y,z]
        %   nonlinear : dimensionless  , [x,y,z,2] as [cos(phi), sin(phi)]
        % The measurement is transformed to match in fit() (section 2.2).
        %
        % The (1-mask) multiply is what enforces the PDF support constraint:
        % only sources outside the ROI are allowed to contribute.

            mask_b  = 1 - mask; % outside mask
            chi_b   = mask_b .* pars.chi_b;                             % [ppm], supported outside the ROI

            % dipole convolution in k-space; D is dimensionless so f stays in ppm
            f       = real(gacelleFFT.ifft3s_(extraData.D .* gacelleFFT.fft3s_(chi_b)));

            if fitting.isnonlinear
                f_rad   = this.phi2 .* f;                      % ppm -> rad
                f       = cat(4, cos(f_rad), sin(f_rad));      % [Nx Ny Nz 2]
            end

        end

        %% Utilities

        % build the dipole kernel and cast inputs
        function [ data, mask, extraData ] = prepare_data(this, data, mask, extraData)

            % --------- Step 1: zeropadding ----------
            [ data, mask, extraData ] = this.zeropadding(data, mask, extraData);

            % convert datatype to single
            data        = single(data);
            mask        = mask > 0;

            matrixSize = size(data);

            [kernel,~]  = this.dipole_kernel(matrixSize,this.voxelSize,this.B0dir);

            % dimensionless kernel (|D| <= 2/3), so the forward maps ppm -> ppm.
            % Field-strength and TE scaling live in phi2, not here.
            extraData.D = kernel;
            extraData   = utils.struct2single(extraData);

        end

        function [ data, mask, extraData ] = zeropadding(this, data, mask, extraData)

            [padPre, padPost] = check_padsize(mask, this.gapMinMM, this.voxelSize);

            matrixSize  = size(data,1:3);

            if any([padPre padPost] > 0)
                pf = @(x,v) padarray(padarray(x, [padPre 0], v, 'pre'), [padPost 0], v, 'post');
            
                data              = pf(data,    0);
                mask              = pf(mask,    false);
                if isfield(extraData,'weights');    extraData.weights  = pf(extraData.weights, 0); end
                
                extraData.padPre      = padPre;
                extraData.padPost     = padPost;
                extraData.matrixSize0 = matrixSize;
            end

            extraData.Nvoxel = nnz(mask);

        end

        % recover chi from y and derive the fitted/residual fields
        function s = postprocess(this, s, mask, extraData, f_Hz)

            % convert the estimated chi_b [ppm] back to a field in Hz
            % and subtract to obtain the local field
            backgroundField_Hz  = real(gacelleFFT.ifft3s_(extraData.D .* gacelleFFT.fft3s_(s.chi_b))) .* (this.gyro * this.B0);
            localField_Hz       = f_Hz - backgroundField_Hz;
            s.backgroundField   = backgroundField_Hz;
            s.localField        = localField_Hz;

            % --- crop last ---
            for k = 1:numel(this.modelParams)
                s.(this.modelParams{k}) = crop_padding(s.(this.modelParams{k}), extraData);
            end
            s.backgroundField   = crop_padding(s.backgroundField, extraData);
            s.localField        = crop_padding(s.localField, extraData);
            s.mask              = crop_padding(mask, extraData);

        
        end

    end

    methods(Static)

        %% signal

        function [dipoleKernel,dKComponents] = dipole_kernel(matrixSize,voxelSize,b0dir)
        % function [dipoleKernel,dKComponents] = DipoleKernel(matrixSize,voxelSize,b0dir)
        %
        % Description: Create dipole kernel in k-space with input matrix dimensions
        %              and spatial resolution
        % Input
        % _____
        %   matrixSize        : image matrix size
        %   voxelSize         : spatial resolution of image
        %   b0dir             : static magnetic field direction (optional)
        %
        % Output
        % ______
        %   dipoleKernal      : Dipole kernel (in k-space), dimensionless,
        %                       real-valued and even, so the adjoint equals
        %                       the forward (conj(D) == D)
        %   dKComponents      : dipole kernel components
        %
        % Kwok-shing Chan @ DCCN
        % k.chan@donders.ru.nl
        % Date created: 24 March 2017
        % Date last modified: 27 September 2017
        %

            if nargin<3
                b0dir = [0 0 1];
            end

            % KC: create 3D matrix in k-space
            [ky,kx,kz] = meshgrid(-matrixSize(2)/2:matrixSize(2)/2-1, ...
                                  -matrixSize(1)/2:matrixSize(1)/2-1, ...
                                  -matrixSize(3)/2:matrixSize(3)/2-1);

            % KC: assign k-vectors
            kx = (kx / max(abs(kx(:)))) / voxelSize(1);
            ky = (ky / max(abs(ky(:)))) / voxelSize(2);
            kz = (kz / max(abs(kz(:)))) / voxelSize(3);

            k2 = kx.^2 + ky.^2 + kz.^2;

            % KC: shift the centre of k-space to matrix corners
            % KC: second term represents (cos beta).^2 where beta is the angle between
            %     k-vector and the static B field
            % dipoleKernel = fftshift( 1/3 - (kz ).^2 ./ (k2 + eps) );
            % 260917: correct for B0 direction, b0dir = [x,y,z];
            % dipoleKernel = fftshift( 1/3 - (kx*b0dir(2) + ky*b0dir(1) + kz*b0dir(3)).^2 ./ (k2 + eps) );
            dipoleKernel = fftshift( 1/3 - (kx*b0dir(1) + ky*b0dir(2) + kz*b0dir(3)).^2 ./ (k2 + eps) );

            dKComponents.kx = fftshift(kx);
            dKComponents.ky = fftshift(ky);
            dKComponents.kz = fftshift(kz);
            dKComponents.k2 = dKComponents.kx.^2 + dKComponents.ky.^2 + dKComponents.kz.^2;

        end

        %% Utilities
        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default(fitting)
        % Fixed options and why they are fixed:
        %   enableComplex    = false : the nonlinear branch is real-valued by
        %                              construction (cos/sin stacked), so no
        %                              complex dlarray support is required
        %   isMaskedOut      = false : the loss is evaluated over the ROI, but
        %                              the parameter lives outside it
        %   isOptimiseMemory = false : parameter and loss supports are
        %                              complementary, so parameters cannot be
        %                              compacted onto the loss mask
        %
        % NOTE: tol/convergenceValue are NOT comparable between branches. The
        % linear loss is in ppm^2 and unbounded; the nonlinear loss is
        % dimensionless, bounded in [0,4] per element, and spread over twice
        % as many elements because of the cos/sin stacking.

            % get basic fitting setting check
            if ~isfield(fitting,'solver');      fitting.solver = 'askadam';        end

            % mcmc is not supported: the phasor-stacked fidelity is not a
            % Gaussian likelihood in the measured field
            if strcmpi(fitting.solver,'mcmc')

                warning('Only askadam.m is supported. Switched to askadam.');

                fitting.solver = 'askadam';

            end
            % askadam
            fitting2 = askadam.check_set_default_basic(fitting);
            % get customised fitting setting check
            if ~isfield(fitting,'regmap');      end

            % data fidelity formulation; see class header
            if ~isfield(fitting,'isnonlinear');     fitting2.isnonlinear    = false; end

            fitting2.enableComplex      = false;
            fitting2.isMaskedOut        = false;
            fitting2.isOptimiseMemory   = false;
            fitting2.lambda             = {0};    % askadam.m built-in TV is not support because the inverted mask is needed.
            fitting2.regmap             = {'chi_b'};
            fitting2.TVmode             = '3D';

        end

        function display_algorithm_info(fitting)
            %%%%%%%%%% 3. display some algorithm parameters %%%%%%%%%%
            disp('--------------');
            disp('Fitting option');
            disp('--------------');
            % type of fitting
            if fitting.isnonlinear
                disp('Non-linear PDF');
            else
                disp('Linear PDF');
            end

            disp('------------------------------------');

        end

    end
end