classdef gpumcTFI < handle

    properties (GetAccess = public, SetAccess = protected)
    % ===== MODEL PARAMETER CONTRACT =====
    % M0        : Proton density weighted signal
    % R2star    : R2* in s^-1
    % y         : preconditioned susceptibility; chi = extraData.P .* y [ppm]
    % phi       : per-voxel phase offset [rad]
    %
    % modelParams{k} <-> ub(k) <-> lb(k) <-> startPoint(k)
    % These four arrays MUST stay the same length and index-aligned.
    % Mutate only as a set - never assign into a single element from
    % outside the class, or these will desync.
    %
    % Unlike most other models in the framework there is no solver-
    % conditional 'noise' parameter and no 'step' array here, because
    % mcmc.m is not supported (see check_set_default).
        modelParams    = {'M0';'R2star';  'y'; 'phi'};
        ub              = [  10;    200;  10; 2*pi  ];
        lb              = [   0;    0.1; -10; -2*pi  ];
        startPoint      = [   1;     30;  0;    0  ];
    end

    properties
    % ===== USER-TUNABLE OPTIONS =====
        thres_tkd   = 0.2;      % TKD truncation threshold, used only to build
        thres_R2s   = [1,150];  % [Hz] valid R2* range; outside this, and any
                                % non-finite value, is zeroed before binning

        gapMinMM    = 20; % mm
        Ps          = 40;
        thres_R2s_P = 30;

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
        dTE;            % [s], echo spacing = te(2)-te(1). Derived in the
                        % constructor. Used by the nonlinear branch to
                        % convert ppm -> rad, and passed to gpuPDF.
        phi2;           % [rad per ppm] = 2*pi*gyro*B0*dTE. Derived in the
                        % constructor; do not set directly.
        te;             % [s], full echo time vector, 1 x nTE. Needed for the
                        % R2* fit; must match size(extraData.img,4).
    end

     properties (Constant)
            gyro = 42.57747892;     % [MHz/T]
    end

    methods
        % constructor
        function this = gpumcTFI(voxelSize,B0,B0dir,te)
        % obj = gpuTFI(voxelSize, B0, B0dir, te)
        %
        % Input
        % ----------
        % voxelSize : voxel size [mm], 1x3
        % B0        : main field strength [T]
        % B0dir     : static field direction [x,y,z]
        % te        : echo time vector [s], 1 x nTE. Echo spacing dTE is
        %             taken as te(2)-te(1), so at least two echoes are
        %             needed. REQUIRED whenever R2* is used, i.e. for
        %             precond 'auto'/'emp+r2s', for the CSF mask, and for
        %             fitting.isnonlinear = true.
        %             Omitting it sets te = dTE = 0, which makes phi2 = 0
        %             and renders the nonlinear forward degenerate
        %             (cos->1, sin->0) with no error raised.
        %
        % Output
        % ----------
        % this      : object of a fitting class
        %
            this.voxelSize   = single(voxelSize(:)).';
            this.B0dir       = single(B0dir(:)).';
            this.B0          = single(B0);

            if nargin < 4
                this.te     = 0;
                this.dTE    = 0;
            else
                this.te = single(te);
                this.dTE = single(te(2)-te(1));
            end

            this.thres_R2s_P = this.thres_R2s_P * (this.B0/3);
            
            % rad per ppm: gyro [MHz/T] * B0 [T] = Hz per ppm; * 2*pi*dTE -> rad per ppm
            this.phi2 = 2*pi*this.gyro*this.B0*this.dTE;

        end

        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('===========================');
            disp('Total field inversion (TFI)');
            disp('===========================');

            fprintf('Field strength (T)         : %g\n', this.B0);
            fprintf('B0 direction [x,y,z]       : [%s]\n', num2str(this.B0dir,' %.2f'));
            fprintf('Voxel size (mm)            : [%s]\n', num2str(this.voxelSize,' %.2f'));
            fprintf('Echo times, TE (ms)        : [%s]\n', num2str(this.te(:).'*1e3,' %.2f'));
            fprintf('Echo spacing, dTE (ms)     : %.2f\n', this.dTE*1e3);

            fprintf('\n')

        end

        %% higher-level data fitting functions
        function  [out] = estimate(this, data, mask, extraData, fitting)
        % Perform mcTFI QSM reconstruction based on askAdam
        %
        % Input
        % -----------
        % data      : Complex multi-echo GRE data, [x,y,z,nTE]. This is NOT
        %             a pre-processed field map - no background removal or
        %             phase unwrapping should have been applied. R2*, M0,
        %             the morphology mask, and the default fidelity weights
        %             are all derived from this array directly (there is no
        %             separate extraData.img; use extraData.R2star/.M0/.MG
        %             below to override any of them individually).
        % mask      : 3D signal mask, [x,y,z]. Region over which the data
        %             fidelity is evaluated (typically the head, e.g.
        %             magnitude > 0.15*max). NOTE: this does NOT restrict
        %             where susceptibility is estimated - that is the whole
        %             FOV.
        % extraData : additional data, all optional
        %   .weights: 3D fidelity weights. Defaults to ones if absent, then
        %             replicated across echoes and the cos/sin channels.
        %             Normalise any user-supplied weights to max 1, or tol
        %             stops being portable across datasets.
        %   .R2star : precomputed R2* map [Hz]. Recomputed from 'data' via
        %             a trapezoidal fit if absent.
        %   .M0     : precomputed M0 map. Recomputed from 'data' via the
        %             same trapezoidal fit if absent.
        %   .MG     : morphology/edge mask for TV, [x,y,z,3], 0 at edges.
        %             Derived from 'data' via gradient_mask if absent.
        %   .M2     : zero-reference mask, e.g. ventricular CSF. Derived
        %             via extract_CSF if absent and fitting.lambdaCSF > 0.
        %   .fint   : initial total field estimate [Hz]. If present, this
        %             triggers an internal gpuPDF fit whose result
        %             OVERWRITES extraData.chi_b (see below), regardless of
        %             whether chi_b was also supplied directly.
        %   .chi_b  : initial susceptibility estimate [ppm], used as
        %             y0 = chi_b./P by estimate_prior. Ignored if .fint is
        %             also present (see above). NOTE: previously documented
        %             here as '.chi0' - that field name is never read by
        %             the code; use 'chi_b'.
        % fitting   : fitting algorithm parameters. fitting.lambdaTV and
        %             fitting.lambdaCSF both default to 0 (i.e. both
        %             regularisation terms off) via check_set_default; set
        %             them explicitly to enable TV/CSF-zero-reference
        %             regularisation. Values are means, not sums, and the
        %             fidelity is in ppm^2 - published MEDI/MEDI+0 lambda
        %             values do not transfer directly and must be
        %             recalibrated on your own data.
        %
        % Output
        % -----------
        % out       : output structure contains all estimation results
        %

            % display basic info
            this.display_data_model_info;

            % get all fitting algorithm parameters
            fitting = this.check_set_default(fitting);

            % build the dipole kernel, validate extraData, cast to single
            [ data, mask, extraData ] = this.prepare_data(data, mask, extraData,fitting);

            %%%%%%%%%%%%%%%% Step 2: Memory management %%%%%%%%%%%%%%%%
            % No partition: the dipole convolution and the TV term are
            % both global, so the volume cannot be split along any spatial
            % axis without changing the operator.

            % parameter estimation
            [out] = this.fit(data,mask,fitting,extraData);

            %%%%%%%%%%%%%%%% End Step 2 %%%%%%%%%%%%%%%%

            % save the estimation results if the output filename is provided
            askadam.save_askadam_output(fitting.outputFilename,out)

        end

        % Data fitting function, can be 3D (voxel-based) or 5D (image-based)
        function [out] = fit(this, data, mask, fitting, extraData)
        %
        % Input
        % -----------
        % data      : Complex multi-echo GRE data, [x,y,z,nTE] (see estimate()
        %             for the full extraData/fitting field reference)
        % mask      : 3D signal mask, [x,y,z]
        % fitting   : fitting algorithm parameters
        % extraData : additional data (see estimate())
        %
        % Output
        % -----------
        % out       : output structure
        %
        % Description: askAdam image-based mcTFI fitting (joint M0, R2*,
        % preconditioned susceptibility y, and phase offset phi)
        %
            
            % get image size
            dims = size(data,1:3);

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

            % set initial tarting points
            pars0 = this.determine_x0(data,mask,extraData,fitting) ;

            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w = this.compute_optimisation_weights(data,fitting,extraData); % This is a tailored funtion

            % % 2.2 display optimisation algorithm parameters
            % this.display_algorithm_info(fitting)

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%
            % split data into real and imaginary parts
            data    = cat(5, real(data), imag(data));

            % 2.4 askAdam optimisation main
            modelFWD = @this.FWD;
            regFcn   = @this.regulariser;
            userFcn  = {modelFWD; regFcn};       % Position #1: forward model function; Position #2: regularisation function

            modelInput  = {fitting, extraData};
            regInput    = {fitting, extraData};
            userInput   = {modelInput;regInput};    % Position #1: forward model extra input; Position #2: regularisation extra input
            
            % 2.3 askAdam optimisation main
            out = askadam().optimisation(data, mask, w, pars0, fitting, userFcn, userInput);
            
            % post processing
            out.final   = this.postprocess(out.final, mask,  extraData, fitting);
            out.min     = this.postprocess(out.min, mask,  extraData, fitting);
                    
            disp('The process is completed.')
            
            % clear GPU
            reset(gpuDevice);
            
        end

        function s = FWD(this, pars, fitting, extraData)
        % mcTFI forward model. Real arithmetic throughout - the complex
        % signal is carried as a stacked [real, imag] pair on dim 5, so no
        % complex dlarray is ever created.
        %
        % Output [x,y,z,nTE,2], matching the measurement built in fit().
        %
        % Sign convention: exp(+i*th), consistent with the v2 forward. If
        % chi comes out negated, flip the sign of th here - check against a
        % vein (should be ~ +0.3 ppm) rather than reasoning about it.

            t       = permute(this.te(:), [2 3 4 1]);              % [1,1,1,nTE]

            % susceptibility -> field, in ppm
            chi     = extraData.P .* pars.y;
            f_ppm   = real(gacelleFFT.ifft3s_(extraData.D .* gacelleFFT.fft3s_(chi)));

            s       = cat(5,pars.M0.*exp(-pars.R2star .* t).*cos(2*pi*this.gyro*this.B0.*f_ppm .* t + pars.phi),...
                            pars.M0.*exp(-pars.R2star .* t).*sin(2*pi*this.gyro*this.B0.*f_ppm .* t + pars.phi));

            % s = pars.M0.*exp(-pars.R2star .* t).*exp(1i*(2*pi*this.gyro*this.B0.*f_ppm .* t + pars.phi));

        end

        %% regulariser
        function cost = regulariser(this, pars, fitting, extraData)

            R_CSF   = 0;
            R_TV    = 0;

            % compute spatial TV
            if fitting.lambdaTV > 0
                R_TV = this.reg_tv(pars, extraData);
            end

            % compute referencing regularisation (if specified)
            if fitting.lambdaCSF > 0
                R_CSF   = this.reg_csf0ref(pars, extraData);
            end

            % concatenate all regularisation terms
            cost = fitting.lambdaTV*R_TV + fitting.lambdaCSF*R_CSF;

        end

        function cost = reg_tv(this, pars, extraData)
        % Morphology-masked total variation on chi = P.*y.
        %
        % Acts on chi, not on y: smoothness is a physical statement about
        % susceptibility. Penalising grad(y) would impose a prior scaled by
        % 1/P - nearly unconstrained in tissue, heavily smoothed in air.
        % This is why the built-in TV cannot be used for this model.
        %
        % Three deliberate choices, all deviations from MEDI/Liu:
        %  1. ANISOTROPIC: sum|gx| + sum|gy| + sum|gz|, not
        %     sum sqrt(gx^2+gy^2+gz^2). Cheaper, but it is not rotation
        %     invariant and mildly favours axis-aligned edges.
        %  2. NO epsilon smoothing. abs() is non-differentiable at 0, but
        %     Adam only needs a subgradient and MATLAB returns sign(0) = 0.
        %     The sqrt(.^2 + eps) form exists for Gauss-Newton CG, which
        %     needs a twice differentiable objective. See epsTV0.
        %  3. Normalised by nnz(MG) so lambdaTV does not scale with matrix
        %     size or FOV padding. Note nnz counts over all 3 gradient
        %     components, so the constant differs from a 3D voxel count by
        %     roughly 3x - it folds into lambdaTV either way.
        %
        % CAVEAT: cgrad is a CENTRAL difference, which annihilates the
        % alternating mode (+1,-1,+1,...) - see the note in cgrad. A TV
        % penalty built on it does not suppress checkerboard noise and
        % decouples the odd/even sublattices. Forward differences are the
        % conventional choice for TV. Left as central for now; if residual
        % high-frequency noise appears in chi, this is the first suspect.

            % apply preconditioner
            chi             = extraData.P .* pars.y;

            % compute gradients
            grad_spatial    = MEDI_helper.fgrad(chi, this.voxelSize);

            % % isotropic TV
            % % P^2-scaled numerical conditioning (see epsTV0 above)
            % epsMap          = this.epsTV0 .* (extraData.P.^2);
            % gmag            = sqrt(sum((extraData.MG .* grad_spatial).^2,4) + epsMap);

            % anisotopic TV
            gmag            = abs(extraData.MG .* grad_spatial);
            
            % compute mean to match askadam.m fidelity term
            cost            = sum(gmag(:))./extraData.Nvoxel;

        end

        function cost = reg_csf0ref(this, pars, extraData)
        % CSF zero-reference: penalise susceptibility variation within M2.
        %
        % Fixes the otherwise arbitrary global offset of chi (the dipole
        % kernel is zero at k = 0, so the mean of chi is unconstrained by
        % the data). Acts on chi = P.*y for the same reason as reg_tv.
        %
        % WARNING - chi(M2) is logical indexing on a dlarray. It produces a
        % variable-length output, which is the known failure mode under
        % dlaccelerate: a cached trace can be reused with a mask from a
        % different call. It also requires M2 to stay logical, but
        % utils.struct2single may cast it to single, in which case this
        % line errors ("array indices must be positive integers or
        % logical"). The commented alternative below has neither problem
        % and is mathematically identical.

            % apply preconditioner
            chi     = extraData.P .* pars.y;
            
            % |M2 .* (x-x_csf)|^2, L2
            M2      = extraData.M2;
            chi_ref = mean(chi(M2)); % chi_ref = sum(M2(:).*chi(:)) ./ max(sum(M2(:)), eps('single'));
            d       = M2 .* (chi - chi_ref);

            % compute mean to match askadam.m fidelity term
            cost    = sum(d(:).^2) ./ nnz(M2);

        end

        %% Prior estimation related functions

        function x0 = determine_x0(this,data,mask,extraData,fitting)
        % Starting point for the preconditioned variable y.
        %
        % If an approximate susceptibility map extraData.chi_b is available
        % (supplied directly, or derived internally from extraData.fint via
        % gpuPDF - see estimate()), initialise y0 = chi_b ./ P. This costs
        % nothing and starts the optimiser near the solution. P >= ~1 by
        % construction so the division is safe.
        %
        % For the linear branch a zero start is merely slow. For the
        % nonlinear branch the objective is periodic and non-convex, and the
        % total field spans many wraps, so a zero start is a genuine risk of
        % converging to the wrong wrap.

            disp('---------------');
            disp('Starting points');
            disp('---------------');

            dims = size(data,1:3);

            if ischar(fitting.start)
                switch lower(fitting.start)
                    case 'prior'
                        % using maximum likelihood method to estimate starting points
                        x0 = this.estimate_prior(data,mask, extraData);
    
                    case 'default'
                        % use fixed points
                        fprintf('Using default starting points for all voxels at [%s]: [%s]\n', cell2str(this.modelParams),replace(num2str(this.startPoint(:).',' %.2f'),' ',','));
                        x0 = utils.initialise_x0(dims,this.modelParams,this.startPoint);

                end
            else
                % user defined starting point
                x0 = fitting.start(:);
                fprintf('Using user-defined starting points for all voxels at [%s]: [%s]\n',cell2str(this.modelParams),replace(num2str(x0(:).',' %.2f'),' ',','));
                x0 = utils.initialise_x0(dims,this.modelParams,x0);

            end

            % make sure the input is bounded
            x0 = askadam.set_boundary(x0,fitting.ub,fitting.lb);

            fprintf('Estimation lower bound [%s]: [%s]\n', cell2str(this.modelParams),replace(num2str(fitting.lb(:).',' %.2f'),' ',','));
            fprintf('Estimation upper bound [%s]: [%s]\n', cell2str(this.modelParams),replace(num2str(fitting.ub(:).',' %.2f'),'  ',','));
            disp('---------------');
        end

        % closed-form solution to estimate better starting points
        function pars0 = estimate_prior(this,data, mask, extraData)

            dims = size(data,1:3);

            % initialise starting point
            for k = 1:numel(this.modelParams)
                pars0.(this.modelParams{k}) = single(this.startPoint(k)*ones(dims));
            end

            disp('Estimate starting points using closed-form solutions...')
            
            start   = tic;
            % R2* and M0
            if ~isfield(extraData,'R2star') || ~isfield(extraData,'M0')
                [r2s,M0] = R2star_trapezoidal(abs(data),this.te);

                r2s(~isfinite(r2s))  = 0;
                M0(~isfinite(M0))    = 0;

                pars0.M0        = M0  ;
                pars0.R2star    = r2s ;
            end
            if isfield(extraData,'R2star')
                pars0.R2star    = extraData.R2star;
            end
            if isfield(extraData,'M0')
                pars0.M0        = extraData.M0;
            end

            % phi
            phi0        = data(:,:,:,1) .* conj(exp(1i*2*pi*extraData.fint.*this.te(1)));
            phi         = polyfit3D_NthOrder(double(angle(phi0)),mask,4) .* mask;
            pars0.phi   = single(phi);

            if isfield(extraData,'chi_b')
                pars0.y     = extraData.chi_b ./ extraData.P;
            end
            
            ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
            fprintf('Starting points estimated. Elapsed time (hh:mm:ss): %s \n',string(ET));
  
        end

        %% Utilities

        function [ data, mask, extraData ] = prepare_data(this, data, mask, extraData, fitting)

            % --------- Step 1: zeropadding ----------
            [ data, mask, extraData ] = this.zeropadding(data, mask, extraData);

            % --------- Step 2: normalise data ----------
            [scale, data] = this.mctfi_data_scaling(data, mask);

            [r2s,M0] = R2star_trapezoidal(abs(data),this.te); % quick R2* mapping
            % scale = prctile(M0(and(mask,isfinite(M0))),95);
            % data  = data./scale;

            r2s(~isfinite(r2s)) = 0; r2s(r2s<0) = 0;
            M0(~isfinite(M0))   = 0; M0(M0<0)   = 0;

            extraData.scale     = scale;
            extraData.R2star    = r2s;
            extraData.M0        = M0;

            % ---------- Step 3: pre-compute dipole kernel ----------
            % dimensionless kernel (|D| <= 2/3), so the forward maps ppm -> ppm.
            % Field-strength and TE scaling live in phi2, not here.
            extraData.D = dipole_kernel(size(data,1:3),this.voxelSize,this.B0dir);

            % ---------- Step 4: zero-reference region ----------
            if fitting.lambdaCSF > 0 && (~isfield(extraData,'M2') || isempty(extraData.M2))
                extraData.M2    = MEDI_helper.extract_CSF(extraData.R2star, mask, this.voxelSize);
            end

            % ---------- Step 5: morphology mask defaults to plain TV ----------
            if fitting.lambdaTV > 0 && (~isfield(extraData,'MG') || isempty(extraData.MG))
                extraData.MG    = MEDI_helper.gradient_mask( sqrt(sum(abs(data).^2,4)), mask, this.voxelSize, @MEDI_helper.fgrad) .* mask >0;
                % extraData.MG    = MEDI_helper.gradient_mask( sqrt(sum(abs(data).^2,4)), mask, this.voxelSize, @MEDI_helper.fgrad);    % include background
            end

            if isfield(extraData,'fint')
                objGPU = gpuPDF(this.voxelSize, this.B0, this.B0dir,this.dTE);
                weights_tmp = sum(abs(data).^2,4);
                weights_tmp = weights_tmp ./max(weights_tmp(:));

                extraData_tmp                   = [];
                extraData_tmp.weights           = weights_tmp;
                fitting_tmp                     = [];
                fitting_tmp.tol                 = 1e-6;
                fitting_tmp.initialLearnRate    = 0.01;
                fitting_tmp.decayRate           = 0.001;
                fitting_tmp.convergenceValue    = 1e-5;
                mask_tmp = imdilate(mask,strel('sphere',1));
                evalc("out = objGPU.estimate(extraData.fint,mask_tmp,extraData_tmp,fitting_tmp)"); % suppress verbose output
                extraData.chi_b = out.final.chi_b; % in ppm
                % chi_in      = this.TKD(out.final.localField.*imerode(mask,strel('sphere',3))/this.gyro/this.B0, imerode(mask,strel('sphere',3)), this.voxelSize, this.B0dir, this.thres_tkd);
                % extraData.chi_b(abs(chi_in)>0.15) = chi_in(abs(chi_in)>0.15);
            end

            % ---------- Step 6: compute preconditioner ----------
            extraData.P = MEDI_helper.compute_preconditioner(this.Ps,data,mask, fitting.precond, extraData.R2star.*mask, this.thres_R2s_P);

            % ---------- Step 7: convert datatype to single ----------
            data        = single(data);
            mask        = mask > 0;
            extraData   = utils.struct2single(extraData);

        end

        % recover chi from y and derive the fitted/residual fields
        function s = postprocess(this, s, mask, extraData, fitting)

            % --- everything below runs at PADDED size ---
            chi = extraData.P .* s.y;
        
            % zero referencing
            if fitting.lambdaCSF > 0
                chi = chi - sum(extraData.M2(:).*chi(:)) ./ sum(extraData.M2(:));
            end
        
            fittedField_Hz = real(gacelleFFT.ifft3s_(extraData.D .* gacelleFFT.fft3s_(chi))) ...
                             .* (this.gyro * this.B0);
        
            % --- crop last ---
            for k = 1:numel(this.modelParams)
                s.(this.modelParams{k}) = crop_padding(s.(this.modelParams{k}),            extraData);
            end
            s.chi           = crop_padding(chi,            extraData);
            s.fittedField   = crop_padding(fittedField_Hz, extraData);
            s.P             = crop_padding(extraData.P, extraData);
            s.mask          = crop_padding(mask, extraData);

            s.M0 = s.M0 .* extraData.scale;

            % s.residualField = f_Hz - s.fittedField;      % f_Hz is the ORIGINAL, uncropped input
        
        end

        % compute weights for optimisation
        function w = compute_optimisation_weights(this,data,fitting,extraData)
        %
        % Output
        % ------
        % w         : 3D weights
        %
        % Typically the magnitude image. Normalise to max 1 before passing
        % in: the weight multiplies the residual before squaring, so an
        % arbitrary magnitude scale shifts the loss by that scale squared and
        % makes tol non-portable across datasets.

            if ~isempty(extraData) && isfield(extraData,'weights')
                w = extraData.weights;
            else
                w = ones(size(data));
            end

            if ndims(w) == 3
                w = repmat(w,1,1,1,numel(this.te));
            end

            % replicate along the phasor dimension so the weight applies to
            % both the cos and sin channels of the nonlinear residual
            % if fitting.isnonlinear
            w = repmat(w,1,1,1,1,2);
            % end
        end
        
        function [ data, mask, extraData ] = zeropadding(this, data, mask, extraData)

            [padPre, padPost] = check_padsize(mask, this.gapMinMM, this.voxelSize);

            matrixSize  = size(data,1:3);

            if any([padPre padPost] > 0)
                pf = @(x,v) padarray(padarray(x, [padPre 0], v, 'pre'), [padPost 0], v, 'post');
            
                data              = pf(data,    0);
                mask              = pf(mask,    false);
                if isfield(extraData,'weights');    extraData.weights  = pf(extraData.weights, 0); end
                if isfield(extraData,'M2');         extraData.M2       = pf(extraData.M2, 0); end
                if isfield(extraData,'fint');       extraData.fint     = pf(extraData.fint, 0); end
                
                extraData.padPre      = padPre;
                extraData.padPost     = padPost;
                extraData.matrixSize0 = matrixSize;
            end

            extraData.Nvoxel = nnz(mask);

        end

    end

    methods(Static)

        %% signal
        function chi = TKD(localField,mask,voxelSize,B0dir,thre_tkd)

            matrixSize = size(localField);

            % dipole kernel
            kernel = dipole_kernel(matrixSize,voxelSize,B0dir);
            
            % initiate inverse kernel with zeros
            % TYPO: 'like', matrixSize prototypes on the SIZE VECTOR, not
            % the data, so kernel_inv is always double on the CPU. Should be
            % 'like', localField. Harmless (the result is cast downstream)
            % but it forces a CPU round trip for gpuArray input.
            kernel_inv = zeros(matrixSize, 'like', matrixSize);
            % get the inverse only when value > threshold
            kernel_inv( abs(kernel) > thre_tkd ) = 1 ./ kernel(abs(kernel) > thre_tkd);
            % direct dipole inversion method
            chi = real( ifftn( fftn(localField) .* kernel_inv ) ) .* mask;
        
        end

        %% Utilities

        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default(fitting)
        % Fixed options and why they are fixed:
        %   enableComplex    = false : the nonlinear branch is real-valued by
        %                              construction (cos/sin stacked)
        %   isMaskedOut      = false : chi is estimated over the whole FOV
        %   isOptimiseMemory = false : ditto - parameters cannot be compacted
        %                              onto the fidelity mask
        %
        % Built-in TV is switched off: this model supplies its own
        % regularisation (REG), because the priors act on chi = P.*y rather
        % than on the fitted parameter.
        %
        % NOTE: tol/convergenceValue are NOT comparable between branches. The
        % linear loss is in ppm^2 and unbounded; the nonlinear loss is
        % dimensionless, bounded in [0,4] per element, and spread over twice
        % as many elements because of the cos/sin stacking.
        %
        % NOTE: lambdaTV and lambdaCSF both default to 0 below (both
        % regularisation terms off) if not supplied by the caller.
        %
        % lambdaTV/lambdaCSF are also NOT the published Liu values: the
        % regularisation terms here are means rather than sums, and the
        % fidelity is in ppm^2 rather than the published field units, so
        % both weights must be calibrated on your own data.

            if ~isfield(fitting,'solver');      fitting.solver = 'askadam';        end

            % mcmc is not supported: the phasor-stacked fidelity is not a
            % Gaussian likelihood in the measured field, and the whole-FOV
            % parameter count is impractical for sampling
            if strcmpi(fitting.solver,'mcmc')
                warning('Only askadam.m is supported. Switched to askadam.');
                fitting.solver = 'askadam';
            end

            fitting2 = askadam.check_set_default_basic(fitting);

            if ~isfield(fitting,'lambdaCSF');   fitting2.lambdaCSF   = 0;        end
            if ~isfield(fitting,'lambdaTV');    fitting2.lambdaTV    = 0;        end
            if ~isfield(fitting,'precond');     fitting2.precond    = 'none';        end

            fitting2.enableComplex      = false;
            fitting2.isMaskedOut        = false;
            fitting2.isOptimiseMemory   = false;

        end
    
        function [C, S] = mctfi_data_scaling(S, mask, isVerbose)
        %MCTFI_DATA_SCALING  Global complex-data scaling used by mcTFI.
        %
        %   C        = mctfi_data_scaling(S, mask)
        %   [C, Sn]  = mctfi_data_scaling(S, mask)
        %   [...]    = mctfi_data_scaling(S, mask, isVerbose)
        %
        % Implements the scaling of Wen et al, Magn Reson Med 2021;86:2165-2178,
        % Section 2.2, item 1:
        %
        %   C = (1/#TE) * sum_j [ (1/#M) * sum_{k in M} |S_{j,k}| ]
        %
        % chosen so that the scaled data satisfies
        %
        %   (1/#TE) * sum_j (1/#M) * sum_{k in M} |S_{j,k}/C| = 1
        %
        % i.e. the mean magnitude over the ROI and over all echoes is exactly 1.
        %
        % Input
        % -----
        %   S         : complex multi-echo GRE, [x,y,z,nTE]. Magnitude-only input
        %               is accepted but then the scaled output is magnitude too.
        %   mask      : 3D ROI mask (logical or numeric). Must be the UNPADDED
        %               mask - see note below.
        %   isVerbose : print a short report. Default true.
        %
        % Output
        % ------
        %   C         : scalar scaling factor, in the units of S
        %   Sn        : S ./ C (only computed if requested)
        %
        % WHY THIS PARTICULAR SCALING
        %   Any global scaling leaves the argmin over chi, R2* and phi0 unchanged -
        %   S -> alpha*S simply scales m0 and the residual by alpha. What it fixes
        %   is the SCALE of the cost function, and therefore the meaning of the
        %   regularisation weights.
        %
        %   In nTFI the noise weighting w is normalised so its mean over the brain
        %   mask is 1. mcTFI has no w at all (Eq. 8 vs Eq. 6), so this data scaling
        %   plays the equivalent role: it is what makes the published lambda values
        %   transferable. Use a different convention - a percentile, a max, an RMS -
        %   and the lambda conversion below no longer holds.
        %
        % REGULARISATION WEIGHTS
        %   With this scaling in place, the paper's conversion from the nTFI values
        %   (lambda1 = 0.001, lambda2 = 0.1) is
        %
        %       lambda1_mcTFI = lambda1_TFI * sum_j t_j
        %       lambda2_mcTFI = lambda2_TFI * sum_j t_j^2
        %
        %   with t_j in seconds. The factors arise because mcTFI sums the fidelity
        %   over echoes while nTFI has a single field term, and each echo's
        %   sensitivity to chi scales with t_j.
        %
        %   NOTE these assume the regularisation terms are plain SUMS, as in the
        %   paper. If your implementation normalises them by an element count
        %   (means rather than sums), multiply back by that count or the transfer
        %   is broken.
        %
        % NOTES
        %   - Plain mean of the MAGNITUDE, not RMS and not a percentile. It is more
        %     sensitive to how much of the ROI is low-signal tissue than a
        %     percentile would be, which is the intended behaviour: it tracks the
        %     average signal level the fidelity term actually sees.
        %   - Compute this BEFORE zero padding. Padded voxels sit outside the mask
        %     so they would not contribute anyway, but computing it first keeps the
        %     value independent of the padding choice.
        %   - Every echo is weighted equally, so later (lower-SNR) echoes pull C
        %     down. That is deliberate - it is the mean over the whole dataset the
        %     cost function sums over.
        
            if nargin < 3 || isempty(isVerbose); isVerbose = true; end
        
            matrixSize = size(S,1:3);
            nTE        = size(S,4);
        
            mask = mask > 0;
            if ~isequal(size(mask,1:3), matrixSize)
                error('mctfi_data_scaling:sizeMismatch', ...
                      'mask (%s) must match the first three dimensions of S (%s).', ...
                      mat2str(size(mask,1:3)), mat2str(matrixSize));
            end
            if ~any(mask(:))
                error('mctfi_data_scaling:emptyMask','Mask is empty.');
            end
        
            % magnitude, flattened to [nVoxel, nTE]
            mag = reshape(abs(S), [], nTE);
            mag = mag(mask(:), :);
        
            % exclude non-finite entries rather than letting them poison the mean
            bad = ~isfinite(mag);
            if any(bad(:))
                warning('mctfi_data_scaling:nonFinite', ...
                        '%d non-finite magnitude values in the ROI were excluded.', nnz(bad));
                mag(bad) = NaN;
                Cj = mean(mag, 1, 'omitnan');
            else
                Cj = mean(mag, 1);
            end
        
            C = mean(Cj);
        
            if ~isfinite(C) || C <= 0
                error('mctfi_data_scaling:degenerate', ...
                      'Scaling factor is %g. Check the mask and the input data.', C);
            end
        
            C = cast(C, 'like', real(S(1)));
        
            if isVerbose
                fprintf('---------------------------------------------\n');
                fprintf('mcTFI data scaling\n');
                fprintf('---------------------------------------------\n');
                fprintf('ROI voxels           : %d\n', nnz(mask));
                fprintf('Echoes               : %d\n', nTE);
                fprintf('Per-echo mean |S|    : [%s]\n', num2str(Cj, '%.3g '));
                fprintf('Scaling factor C     : %.6g\n', C);
                fprintf('Mean |S/C| over ROI  : %.4f  (should be 1.0000)\n', mean(Cj)/C);
                fprintf('---------------------------------------------\n');
            end
        
            if nargout > 1
                S = S ./ C;
            end
        
        end
    end

end