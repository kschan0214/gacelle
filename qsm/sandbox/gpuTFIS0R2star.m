classdef gpuTFIS0R2star < handle
% Preconditioned Total Field Inversion (TFI) for QSM.
%
% Estimates the susceptibility distribution over the ENTIRE field of view
% directly from the total field, without a separate background field removal
% step. Background and local sources are estimated jointly, which avoids the
% error propagation inherent to the sequential (remove-then-invert) pipeline.
%
%   chi = P .* y
%   y*  = argmin_y  1/2*|| w (f - d*(P.*y)) ||_2^2
%                 + lambdaTV *mean( | MG .* grad(P.*y) | )
%                 + lambdaCSF*mean( | M2 .* (P.*y - mean_{M2}(P.*y)) |^2 )
%
% Both regularisation terms are normalised by their own element count so
% that the weights are portable across matrix size, FOV padding and
% subject-specific ROI volume. This differs from the published formulation,
% which uses plain sums; the published lambda values therefore do NOT
% transfer directly.
%
% The fitted variable is y, NOT chi. P whitens the solution: it is chosen
% proportional to the expected susceptibility magnitude at each voxel, so
% that y has roughly unit variance everywhere and a single learning rate is
% appropriate across soft tissue, bone and air. chi is recovered on output.
%
% P is built internally according to fitting.precond:
%   'auto'      Liu 2020 adaptive preconditioner. Runs gpuPDF + TKD to get a
%               rough chi estimate, then bins |chi| against R2* inside the
%               ROI (sigmoid fit) and against distance-to-ROI outside
%               (cubic decay fit). Requires extraData.img.
%   'emp+r2s'   Liu 2017 empirical preconditioner with R2*: P = Ps outside
%               the ROI and at high-R2* voxels (haemorrhage/calcification),
%               1 elsewhere.
%   'empirical' Liu 2017 binary preconditioner: P = Ps outside the ROI, 1
%               inside.
%   'none'      P = 1. Unpreconditioned TFI. See the bounds note below -
%               the default bounds are NOT valid in this mode.
%
% References
%   Liu Z et al. Preconditioned total field inversion (TFI) method for
%     quantitative susceptibility mapping. MRM 2017;78:303-315.
%   Liu Z et al. Automated adaptive preconditioner for quantitative
%     susceptibility mapping. MRM 2020;83:271-285.
%   Liu T et al. Morphology enabled dipole inversion (MEDI). NeuroImage
%     2012;59:2560-2568.   (MG construction, gradient_mask)
%
% SUPPORT CONVENTIONS (differs from gpuPDF - read before use)
%   parameter support : the whole FOV. Every voxel carries a susceptibility,
%                       including air, bone and scalp. This is the point of
%                       TFI. isOptimiseMemory must stay false.
%   loss support      : wherever the field measurement is trustworthy,
%                       i.e. the head/signal mask. Passed as `mask`.
%   MG                : morphology (edge) mask for the TV term. Following
%                       MEDI convention this is 0 AT tissue edges and 1
%                       elsewhere, so susceptibility edges are permitted
%                       where the magnitude image has one. It is [x,y,z,3] -
%                       one mask per gradient direction - and is zeroed
%                       outside the ROI, so the TV term does NOT constrain
%                       air, bone or scalp. Those voxels are determined by
%                       data fidelity alone, through the nonlocal dipole
%                       coupling to the measured interior field.
%   M2                : zero-reference region, typically ventricular CSF.
%                       Set fitting.lambdaCSF = 0 to disable (e.g. cardiac).
%
% REQUIRED extraData
%   .img              : complex or magnitude multi-echo GRE, [x,y,z,nTE].
%                       Used to derive R2*, MG, the fidelity weights and
%                       (for precond 'auto') the initial susceptibility
%                       estimate. Removed from extraData before fitting.
%
% Two data fidelity formulations, via fitting.isnonlinear:
%   false (default) : residual on the field itself, in ppm. Matches the
%                     published TFI formulation.
%   true            : residual between unit phasors, implemented as a
%                     stacked [cos, sin] pair so the built-in L2 loss gives
%                     |exp(ia)-exp(ib)|^2 = 2(1-cos(a-b)). Requires dTE.
%                     See the warning in check_set_default before using it.
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
    % y     : preconditioned susceptibility [ppm / P], dimensionless-ish
    %
    % modelParams{k} <-> ub(k) <-> lb(k) <-> startPoint(k) <-> step(k)
    % These five arrays MUST stay the same length and index-aligned.
    % Mutate only as a set, via updateProperty() - never assign into a
    % single element from outside the class, or these will desync.
    %
    % No solver-conditional 'noise' parameter: mcmc.m is not supported.
    % Any future solver-conditional parameter should be appended LAST.
    %
    % Bounds are on y, NOT chi, and they assume preconditioning is active.
    % With a well-formed P, y sits within roughly +/-1 everywhere
    % (tissue: chi ~ 0.2 ppm, P ~ 1; air: chi ~ 9.4 ppm, P ~ Ps).
    %
    % WARNING: with fitting.precond = 'none' (P == 1) the bounds are wrong -
    % air is ~9.4 ppm and the fit will silently saturate at +/-1. Widen them
    % to at least +/-12 for that mode, or do not use it.
    %
    % If the fit pins at the bound with preconditioning on, suspect P.
        modelParams     = { 'S0'; 'R2s'; 'y'; 'phi'  };
        ub              = [    2;   300;  10; -2*pi  ];
        lb              = [    0;   0.1; -10;  2*pi  ];
        startPoint      = [    1;    30;   0;   0    ];
    end

    properties
    % ===== USER-TUNABLE OPTIONS =====

        thres_tkd   = 0.2;      % TKD truncation threshold, used only to build
                                % the rough chi estimate for precond 'auto'
        thres_R2s   = [1,150];  % [Hz] valid R2* range; outside this, and any
                                % non-finite value, is zeroed before binning
        Ps          = 40;       % preconditioning weight for strong sources
                                % (background, and high-R2* voxels in
                                % 'emp+r2s'). Also caps P in 'auto' mode.

        % CURRENTLY UNUSED - retained for reference.
        % Liu 2020 replaces |MG grad(chi)| with sqrt(|MG grad(chi)|^2 + eps),
        % scaling eps with P^2, because Gauss-Newton CG needs a twice
        % differentiable objective. askadam.m uses subgradients, so the
        % smoothing is unnecessary here and abs() is used directly in
        % reg_tv. Reinstate only if switching to a CG-type solver.
        epsTV0  = 1e-6;

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
        function this = gpuTFIS0R2star(voxelSize,B0,B0dir,te)
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
                this.te = single(te(:));
                this.dTE = single(te(2)-te(1));
            end
            

            % rad per ppm: gyro [MHz/T] * B0 [T] = Hz per ppm; * 2*pi*dTE -> rad per ppm
            this.phi2 = 2*pi*this.gyro*this.B0*this.dTE;

        end

        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('============================================');
            disp('Total field inversion (TFI) with R2* mapping');
            disp('============================================');

            fprintf('Field strength (T)         : %g\n', this.B0);
            fprintf('B0 direction [x,y,z]       : [%s]\n', num2str(this.B0dir,' %.2f'));
            fprintf('Voxel size (mm)            : [%s]\n', num2str(this.voxelSize,' %.2f'));
            fprintf('Echo times, TE (ms)        : [%s]\n', num2str(this.te(:).'*1e3,' %.2f'));
            fprintf('Echo spacing, dTE (ms)     : %.2f\n', this.dTE*1e3);

            fprintf('\n')

        end

        %% higher-level data fitting functions
        function  [out] = estimate(this, data, mask, extraData, fitting)
        % Perform TFI QSM reconstruction based on askAdam
        %
        % Input
        % -----------
        % data      : 3D TOTAL field map in Hz, [x,y,z]. Not a local field -
        %             no background removal should have been applied.
        % mask      : 3D signal mask, [x,y,z]. Region over which the data
        %             fidelity is evaluated (typically the head, e.g.
        %             magnitude > 0.15*max). NOTE: this does NOT restrict
        %             where susceptibility is estimated - that is the whole
        %             FOV.
        % extraData : additional data
        %   .img    : REQUIRED. Multi-echo GRE, [x,y,z,nTE]. Source for R2*,
        %             MG, the fidelity weights and (precond 'auto') chi_est.
        %             Deleted from extraData before fitting.
        %   .MG     : optional. Morphology/edge mask for TV, [x,y,z,3],
        %             0 at edges. Derived from .img via gradient_mask if
        %             absent.
        %   .M2     : optional. Zero-reference mask, e.g. ventricular CSF.
        %             Derived via extract_CSF if absent and lambdaCSF > 0.
        %   .R2s    : optional. Precomputed R2* map [Hz]. Recomputed from
        %             .img if absent.
        %   .weights: optional. 3D fidelity weights. Defaults to the
        %             echo-summed magnitude, normalised to max 1. Normalise
        %             any user-supplied weights the same way, or tol stops
        %             being portable across datasets.
        %   .chi0   : optional initial susceptibility estimate [ppm], used
        %             as y0 = chi0./P. NOTE: precond 'auto' already computes
        %             a suitable chi_est internally but currently discards
        %             it - see compute_adaptive_preconditioner.
        % fitting   : fitting algorithm parameters. Must include lambdaTV
        %             and lambdaCSF; neither has a default (see
        %             check_set_default).
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

        % Data fitting function, image-based (3D)
        function [out] = fit(this,data,mask,fitting,extraData)
        %
        % Input
        % -----------
        % f_Hz      : 3D total field map [Hz], [x,y,z]
        % mask      : 3D signal mask, [x,y,z] (fidelity support only)
        % fitting   : fitting algorithm parameters
        % extraData : see estimate
        %
        % Output
        % -----------
        % out       : output structure
        %   .final      : final results
        %       .y              : preconditioned variable (fitted quantity)
        %       .chi            : susceptibility [ppm] = P .* y  <-- USE THIS
        %       .fittedField    : d*chi, in Hz
        %       .residualField  : f_Hz - fittedField, in Hz
        %       .loss           : final loss metric
        %   .min        : as above, at the iteration with minimum loss
        %
        % Units: chi is returned in ppm. Fields are returned in Hz, i.e. the
        % same units as the input, so no conversion is needed by the caller.
        % If a zero-reference mask M2 was supplied, chi is additionally
        % referenced to the mean within it.
        %

            % get image size
            dims = size(data,1:3);

            %%%%%%%%%%%%%%%%%%%% 1. Validate and parse input %%%%%%%%%%%%%%%%%%%%
            if nargin < 3 || isempty(mask); mask = ones(dims,'logical'); end
            if nargin < 4; fitting = struct(); end

            % get all fitting algorithm parameters
            fitting                 = this.check_set_default(fitting);
            % determine fitting parameters
            fitting.modelParams     = this.modelParams;
            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(this.modelParams)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(this.modelParams)); end

            % set initial starting points
            pars0                   = this.determine_x0(data,fitting,mask);

            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w = this.compute_optimisation_weights(extraData,data,fitting);

            % split data into real and imaginary parts for complex-valued data
            data = cat(5,real(data),imag(data));

            % 2.3 display optimisation algorithm parameters
            this.display_algorithm_info(fitting)

            % 2.4 askAdam optimisation main
            modelFWD = @this.FWD;
            regFcn   = @this.regulariser;
            userFcn  = {modelFWD; regFcn};       % Position #1: forward model function; Position #2: regularisation function

            modelInput  = { extraData};
            regInput    = {mask, fitting, extraData};
            userInput   = {modelInput;regInput};    % Position #1: forward model extra input; Position #2: regularisation extra input
            
            out = askadam().optimisation(data, mask, w, pars0, fitting, userFcn, userInput);
            % out = askadam().optimisation(f, mask, w, pars0, fitting, @this.FWD, mask, fitting, extraData);

            out.final = this.postprocess(out.final, data, extraData, fitting);
            out.min   = this.postprocess(out.min,   data, extraData, fitting);


            disp('The process is completed.')
            disp('##############################################')

            % clear GPU
            reset(gpuDevice)

        end

        % recover chi from y and derive the fitted/residual fields
        function s = postprocess(this, s, data, extraData, fitting)
        % NOTE: chi is referenced to the M2 mean here, in addition to the
        % homogeneity penalty applied during fitting. The two are
        % complementary - the penalty enforces flatness, this sets the
        % absolute level - but it means s.chi is NOT exactly P.*s.y.
        % Anyone recomputing chi from y will find a constant offset.

            chi = extraData.P .* s.y;                                   % [ppm]

            % zero-referencing to the mean within M2, consistent with the
            % regularisation term. Skipped if no reference region was given.
            if isfield(extraData,'M2') && ~isempty(extraData.M2) && fitting.lambdaCSF > 0
                chi = chi - sum(extraData.M2(:).*chi(:)) ./ sum(extraData.M2(:));
            end

            fittedField_Hz  = real(gacelleFFT.ifft3s_(extraData.D .* gacelleFFT.fft3s_(chi))) .* (this.gyro * this.B0);

            s.chi           = chi;
            s.fittedField   = fittedField_Hz;
            s.residualField = data - fittedField_Hz;

        end

        % compute weights for optimisation
        function w = compute_optimisation_weights(this,extraData,data,fitting)
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

        %% Prior estimation related functions

        function x0 = determine_x0(this,y,fitting,mask)
        % Starting point for the preconditioned variable y.
        %
        % If an approximate susceptibility map chi0 is available (it is, if
        % the preconditioner was built the Liu 2020 way, since that pipeline
        % produces chi_est en route), initialise y0 = chi0 ./ P. This costs
        % nothing and starts the optimiser near the solution. P >= ~1 by
        % construction so the division is safe, but it is guarded anyway.
        %
        % For the linear branch a zero start is merely slow. For the
        % nonlinear branch the objective is periodic and non-convex, and the
        % total field spans many wraps, so a zero start is a genuine risk of
        % converging to the wrong wrap.

            disp('---------------');
            disp('Starting points');
            disp('---------------');

            dims = size(y,1:3);

            if ischar(fitting.start)
                switch lower(fitting.start)
                    case 'prior'
                        % using maximum likelihood method to estimate starting points
                        x0 = this.estimate_prior(y,mask);
    
                    case 'default'
                        % use fixed points
                        fprintf('Using default starting points for all voxels at [%s]: [%s]\n', cell2str(this.modelParams),replace(num2str(this.startPoint(:).',' %.2f'),' ',','));
                        x0 = utils.initialise_x0(dims,this.modelParams,this.startPoint);

                end
            else
                % user defined starting point
                x0 = fitting.start(:);
                fprintf('Using user-defined starting points for all voxels at [%s]: [%s]\n',cell2str(this.modelParams),replace(num2str(x0(:).',' %.2f'),' ',','));
                x0 = utils.initialise_x0(dims,this.modelParams,this.startPoint);

            end

            % make sure the input is bounded
            x0 = askadam.set_boundary(x0,fitting.ub,fitting.lb);

            fprintf('Estimation lower bound [%s]: [%s]\n', cell2str(this.modelParams),replace(num2str(fitting.lb(:).',' %.2f'),' ',','));
            fprintf('Estimation upper bound [%s]: [%s]\n', cell2str(this.modelParams),replace(num2str(fitting.ub(:).',' %.2f'),'  ',','));
            disp('---------------');
        end

        % closed-form solution to estimate better starting points
        function pars0 = estimate_prior(this,data,mask)

            dims = size(data,1:3);

            for k = 1:numel(this.modelParams)
                pars0.(this.modelParams{k}) = single(this.startPoint(k)*ones(dims));
            end

            disp('Estimate starting points using closed-form solutions...')
            
            start = tic;
            % R2* closed-form solution
            R2s0                    = this.R2star_trapezoidal(abs(data),this.te);
            R2s0(~isfinite(R2s0))   = this.lb(2); R2s0(R2s0<this.lb(2)) = this.lb(2); R2s0(R2s0>this.ub(2)) = this.ub(2);

            % always follow the order specified in the beginning of the file
            pars0.(this.modelParams{2}) = single(R2s0);

            S0 = abs(data(:,:,:,1)) .* exp(R2s0*this.te(1));
            pars0.(this.modelParams{1}) = single(S0);

            %% phi
            dt   = mean(diff(this.te));
            H    = sum( data(:,:,:,2:end) .* conj(data(:,:,:,1:end-1)), 4 );
            f_Hz = angle(H) ./ (2*pi*dt);                  % check sign against your convention
        
            phi0   = data(:,:,:,1) .* exp(-1i*2*pi*f_Hz*this.te(1)) .* exp(R2s0*this.te(1));
            phi0   = PolyFit(angle(phi0),mask,2);
            pars0.(this.modelParams{4}) = single(phi0);

            ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
            fprintf('Starting points estimated. Elapsed time (hh:mm:ss): %s \n',string(ET));
  
        end

        %% Signal related functions

        % compute the forward model
        function s = FWD(this, pars, extraData)

            t = permute(this.te,[2 3 4 1]);

            % ---- field from susceptibility ----
            chi   = extraData.P .* pars.y;                                      % ppm, apply preconditioner
            f_ppm = real(gacelleFFT.ifft3s_(extraData.D .* gacelleFFT.fft3s_(chi)));

            Sr = pars.S0 .*exp(-pars.R2s .* t) .* cos(2*pi*f_ppm .* (this.gyro * this.B0) .* t + pars.phi);
            Si = pars.S0 .*exp(-pars.R2s .* t) .* sin(2*pi*f_ppm .* (this.gyro * this.B0) .* t + pars.phi);

            % ---- prediction, stacked real ----
            s = cat(5, Sr, Si);                % [x,y,z,nTE,2]
        
            % % ---- unit-amplitude template, real form ----
            % TE    = permute(this.te(:),[ 2 3 4 1]);                        % [1,1,1,nTE]
            % decay = exp(-pars.R2s .* TE);
            % th    = -2*pi .* f_ppm .* (this.gyro * this.B0) .* TE;       % sign convention: e^{-i*th}
            % gr    =  decay .* cos(th);
            % gi    = -decay .* sin(th);
            % 
            % % ---- VARPRO amplitude, analytic ----
            % den   = sum(decay.^2, 4) + eps('single');                   % = sum(gr^2+gi^2), theta-free
            % Ar    = sum(extraData.Sr.*gr + extraData.Si.*gi, 4) ./ den;
            % Ai    = sum(extraData.Si.*gr - extraData.Sr.*gi, 4) ./ den;
            % 
            % % ---- prediction, stacked real ----
            % s = cat(5, Ar.*gr - Ai.*gi, ...
            %            Ar.*gi + Ai.*gr);                % [x,y,z,nTE,2]
        end
        
        % regularisation
        function cost = reg_tv(this, pars, fitting, extraData)
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
            grad_spatial    = this.fgrad(chi, this.voxelSize);

            switch lower(fitting.TVmode)
                case 'isotropic'
                    % isotropic TV
                    % P^2-scaled numerical conditioning (see epsTV0 above)
                    epsMap          = this.epsTV0 .* (extraData.P.^2);
                    gmag            = sqrt(sum((extraData.MG .* grad_spatial).^2,4) + epsMap);

                case 'anisotropic'

                    % anisotopic TV
                    gmag            = abs(extraData.MG .* grad_spatial);
            end
            
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
            chi = extraData.P .* pars.y;
            
            % |M2 .* (x-x_csf)|^2, L2
            M2      = extraData.M2;
            chi_ref = mean(chi(M2)); % chi_ref = sum(M2(:).*chi(:)) ./ max(sum(M2(:)), eps('single'));
            d       = M2 .* (chi - chi_ref);

            % compute mean to match askadam.m fidelity term
            cost    = sum(d(:).^2) ./ nnz(M2);

        end

        function cost = reg_tnv(this, pars, mask, extraData)
        % Total Nuclear Variation on (chi, R2*), blended to vectorial TV
        % where R2* blooming is expected.
        %
        % Form the 3x2 Jacobian J = [grad(chi_n), grad(R2s_n)] per voxel and
        % penalise its nuclear norm. For two channels this has a closed
        % form, so no SVD is needed:
        %
        %   a = |grad chi|^2, b = |grad R2s|^2, c = grad chi . grad R2s
        %   ||J||_* = sqrt( a + b + 2*sqrt(a*b - c^2) )
        %
        % and since a*b - c^2 = |grad chi|^2 |grad R2s|^2 sin^2(theta),
        %
        %   ||J||_* = sqrt( a + b + 2|grad chi||grad R2s||sin theta| )
        %           = VTV + a misalignment term weighted by the PRODUCT of
        %             the two gradient magnitudes.
        %
        % That product weighting is what makes this better behaved than a
        % normalised (1 - |cos theta|) coupling term:
        %   - flat regions contribute ~0 instead of saturating at maximum
        %     penalty on noise-defined directions;
        %   - a legitimate single-channel edge (chi changes, R2* does not)
        %     costs nothing extra, instead of being penalised at 90 degrees.
        % It is also convex in the gradients, and needs ONE weight rather
        % than three (two per-map TV + one coupling).
        %
        % Minimising the nuclear norm drives J toward rank 1, i.e. the two
        % gradients become parallel. ||J||_* >= ||J||_F with equality
        % exactly when they are.
        %
        % The eps guards are NOT optional: both sqrt calls hit zero wherever
        % MG = 0, which is most of the FOV, and d/dx sqrt(x) at 0 is Inf.

            chi = extraData.P .* pars.y;

            gc = this.fgrad(chi,      this.voxelSize);
            gr = this.fgrad(pars.R2s, this.voxelSize);
        
            a = sum(gc.^2,  4);
            b = sum(gr.^2,  4);
            c = sum(gc.*gr, 4);
        
            detM = max(a.*b - c.^2, 0);      % = |gc|^2|gr|^2 sin^2(theta)
            detM = detM .*mask;
        
            % squared form: no epsilon needed, gradient defined everywhere
            cost = sum(detM(:)) ./ extraData.Nvoxel;

            % chi = extraData.P .* pars.y;
            % 
            % % masked, scale-normalised gradients, [x,y,z,3]
            % gc = extraData.MG .* this.fgrad(chi      ./ extraData.scaleChi, this.voxelSize);
            % gr = extraData.MG .* this.fgrad(pars.R2s ./ extraData.scaleR2s, this.voxelSize);
            % 
            % a = sum(gc.^2,  4);
            % b = sum(gr.^2,  4);
            % c = sum(gc.*gr, 4);
            % 
            % % >= 0 by Cauchy-Schwarz; clamp against round-off
            % detM = max(a.*b - c.^2, 0);
            % 
            % vtv = sqrt(a + b + this.epsTV0);
            % nuc = sqrt(a + b + 2*sqrt(detM + this.epsTV0) + this.epsTV0);
            % 
            % mix = extraData.wTNV .* nuc + (1 - extraData.wTNV) .* vtv;
            % 
            % cost = sum(mix(:)) ./ extraData.Nvoxel;

            % DIAGNOSTIC - gate behind a debug flag before shipping.
            % The third quantity is the misalignment energy. If it stays
            % near zero the structural prior is doing nothing and this is
            % just an expensive VTV.
            % fprintf('a=%.3e b=%.3e misalign=%.3e\n', ...
            %     sum(a(:)), sum(b(:)), sum(sqrt(detM(:))));

        end

        function cost = regulariser(this, pars, mask, fitting, extraData)

            % compute spatial TV
            if fitting.lambdaTV > 0
                R_TV = this.reg_tv(pars, fitting, extraData);
            else
                R_TV = 0;
            end

            % compute referencing regularisation (if specified)
            if fitting.lambdaCSF > 0
                R_CSF   = this.reg_csf0ref(pars, extraData);
            else
                R_CSF   = 0;
            end

            % compute total nuclear variation
            if fitting.lambdaTNV > 0
                R_TNV   = this.reg_tnv(pars, mask, extraData);
            else
                R_TNV   = 0;
            end

            % concatenate all regularisation terms
            cost = fitting.lambdaTV*R_TV + fitting.lambdaCSF*R_CSF +fitting.lambdaTNV*R_TNV;

        end

        % spatial gradients
        function MG = gradient_mask( this, img, mask, voxelSize, grad, percentage)
            % adapted from MEDI toolbox

            if nargin < 6
                percentage = 0.9;
            end
            
            field_noise_level   = 0.01*max(img(:));
            MG                  = abs(grad(img.*(mask>0), voxelSize));
            denominator         = nnz(mask);
            numerator           = sum(MG(:)>field_noise_level);

            if  (numerator/denominator) > percentage
                while (numerator/denominator) > percentage
                    field_noise_level   = field_noise_level*1.05;
                    numerator           = sum(MG(:)>field_noise_level);
                end
            else
                while (numerator/denominator) < percentage
                    field_noise_level   = field_noise_level*.95;
                    numerator           = sum(MG(:)>field_noise_level);
                end
            end
            
            MG = MG <= field_noise_level;
        end

        function Gx = fgrad(this, chi, voxelSize)
        %FGRAD  Discrete gradient using FORWARD differences, Neumann boundary.
        %
        %   Gx = this.fgrad(chi)              gradient of the 3D array chi
        %   Gx = this.fgrad(chi, voxelSize)   scaled by 1/voxelSize(k)
        %
        % Input
        % -----
        %   chi        : 3D array. Also accepts gpuArray and dlarray.
        %   voxelSize  : [dx dy dz], default [1 1 1]
        %
        % Output
        % ------
        %   Gx         : [size(chi) 3], components concatenated along dim 4
        %
        % D(i) = (chi(i+1) - chi(i))/h for i < n, and D(n) = 0. The trailing
        % zero is the Neumann (zero-flux) boundary condition, obtained in the
        % original by padding with the last slice before subtracting.
        %
        % PREFER THIS OVER cgrad FOR TV REGULARISATION.
        % Forward differences have a trivial null space (constants only),
        % whereas the central difference used by cgrad also annihilates the
        % alternating mode (+1,-1,+1,...), so a central-difference TV penalty
        % does not suppress checkerboard noise and decouples the odd and even
        % sublattices. cgrad remains the right choice for deriving an edge
        % mask from a magnitude image, where the half-voxel shift of a
        % forward difference would misregister edges against the grid.
        %
        % The adjoint of this operator is the negative backward divergence
        % (bdiv in the MEDI toolbox). It is not needed here - dlgradient
        % supplies it - but an exactly matched adjoint is required if this
        % operator is ever reused inside a CG-type solver.
        %
        % Original: Youngwook Kee (Oct 2015), MEDI toolbox.
        % References
        %   [1] Chambolle. An Algorithm for Total Variation Minimization and
        %       Applications. JMIV 2004.
        %   [2] Pock et al. Global Solutions of Variational Models with
        %       Convex Regularization. SIIMS 2010.

            if nargin < 3 || isempty(voxelSize)
                voxelSize = [1 1 1];
            end
            if numel(voxelSize) < 3
                voxelSize = [voxelSize(:).' ones(1, 3-numel(voxelSize))];
            end

            Gx = cat(4, this.diff_forward(chi, 1, voxelSize(1)), ...
                        this.diff_forward(chi, 2, voxelSize(2)), ...
                        this.diff_forward(chi, 3, voxelSize(3)));

        end

        function D = diff_forward(this, x, dim, h)
        % Forward difference along `dim`, zero at the trailing slice.
        %
        % Built by slicing and concatenation rather than by padding then
        % subtracting: the original materialised a full shifted copy of chi
        % per dimension before the subtraction, this allocates only the
        % result. Fixed-index slicing traces cleanly under dlarray and
        % dlaccelerate, and mirrors diff_central so the two stay comparable.

            n = size(x, dim);

            if n < 2
                D = x * 0;                  % *0 rather than zeros(...,'like',x)
                return                      % to preserve dlarray/gpuArray type
            end

            idx = repmat({':'}, 1, max(ndims(x), 3));

            interior = this.slice(x, idx, dim, 2:n) - this.slice(x, idx, dim, 1:n-1);
            trailing = this.slice(x, idx, dim, n) * 0;      % Neumann BC

            D = cat(dim, interior, trailing) / h;

        end

        function Gx = cgrad(this, x, voxelSize)
        %CGRAD  Central-difference gradient with one-sided differences at the edges.
        %
        %   Gx = cgrad(x)                returns the gradient of the 3D array x
        %   Gx = cgrad(x, voxel_size)    scales each component by 1/voxel_size(k)
        %
        % Input
        % -----
        %   x           : 3D array. Also accepts gpuArray and dlarray.
        %   voxel_size  : [dx dy dz], default [1 1 1]
        %
        % Output
        % ------
        %   Gx          : [size(x) 3], components concatenated along dim 4
        %
        % Interior points use the second-order central difference
        % (x(i+1)-x(i-1))/2h. The first and last slice along each dimension use
        % first-order one-sided differences, so accuracy drops at the boundary.
        % Dimensions of length 1 return zero; length 2 returns the one-sided
        % difference at both slices.
        %
        % NOTE ON USE IN TV REGULARISATION
        % Central differences have a null space: the highest-frequency alternating
        % mode (+1,-1,+1,...) is annihilated, because x(i+1)-x(i-1) = 0 for it. A
        % TV penalty built on this operator therefore does not penalise checkerboard
        % noise at all, and it decouples the odd and even sublattices. Forward
        % differences (fgrad) are the usual choice for TV; cgrad is appropriate
        % where an unbiased, non-shifted gradient estimate is wanted, e.g. deriving
        % a morphology/edge mask from a magnitude image.
        
            if nargin < 2 || isempty(voxelSize)
                voxelSize = [1 1 1];
            end
            if numel(voxelSize) < 3
                voxelSize = [voxelSize(:).' ones(1, 3-numel(voxelSize))];
            end
        
            Gx = cat(4, this.diff_central(x, 1, voxelSize(1)), ...
                        this.diff_central(x, 2, voxelSize(2)), ...
                        this.diff_central(x, 3, voxelSize(3)));
        
        end
        
        function D = diff_central(this, x, dim, h)
        % Central difference along `dim`, one-sided at the two boundary slices.
        % Built by concatenation rather than indexed assignment: fixed-index
        % slicing traces cleanly under dlarray/dlaccelerate, and there is one
        % allocation instead of a shift plus two in-place writes.
        
            n = size(x, dim);
        
            if n < 2
                D = x * 0;                  % *0 rather than zeros(...,'like',x) to
                return                      % preserve dlarray/gpuArray type safely
            end
        
            idx = repmat({':'}, 1, max(ndims(x), 3));
        
            % leading slice: forward difference
            lo = this.slice(x, idx, dim, 2) - this.slice(x, idx, dim, 1);
            % trailing slice: backward difference
            hi = this.slice(x, idx, dim, n) - this.slice(x, idx, dim, n-1);
        
            if n == 2
                D = cat(dim, lo, hi) / h;
            else
                mid = 0.5 * (this.slice(x, idx, dim, 3:n) - this.slice(x, idx, dim, 1:n-2));
                D   = cat(dim, lo, mid, hi) / h;
            end
        
        end

        % preconditioner
        function P = compute_adaptive_preconditioner(this, f, img, M)
        % Automated adaptive preconditioner (Liu 2020).
        %
        % Input
        % -----
        %   f    : total field map [Hz], full FOV
        %   img  : multi-echo GRE [x,y,z,nTE], for R2* and fidelity weights
        %   M    : binary tissue ROI mask
        %
        % Output
        % ------
        %   P    : preconditioner, dimensionless, in [1, Ps]
        %
        % Steps
        %   1. Rough chi estimate: gpuPDF gives background sources outside M
        %      directly, and its local field is inverted by TKD inside M.
        %      Both are crude by design - the trend fits below only need the
        %      right order of magnitude per bin.
        %   2a. Inside M, bin |chi| by R2* (1 Hz bins), take the median per
        %      bin, fit a sigmoid. Encodes the prior that higher R2* implies
        %      larger |chi| (haemorrhage positive, calcification negative).
        %   2b. Outside M, bin by distance to the ROI boundary, fit a cubic
        %      decay - the far field of a dipole distribution falls off
        %      faster than any local source.
        %   3. Normalise by sigma1 so soft tissue maps to P = 1, then clamp
        %      to [1, Ps].
        %
        % Interpretation: treating chi as N(0, sigma^2(R2*)) voxelwise and
        % using sigma as the weight makes P^-1 chi approximately unit
        % variance, which is what accelerates convergence.
        %
        % NOTE: chi_est computed here is exactly what determine_x0 wants as
        % chi0, but is currently discarded. Returning it would make the
        % starting point free.

            fprintf('Computing automatic preconditioner...')
        
            R2s = this.R2star_trapezoidal(abs(img),this.te);
            bad = R2s > this.thres_R2s(2) | R2s < this.thres_R2s(1) | isnan(R2s) | isinf(R2s);
            R2s(bad) = 0;

            % --- Step 1: fast approximate susceptibility map ---

            weights = sum(abs(img).^2,4);
            weights = weights ./max(weights(:));
            
            objGPU = gpuPDF(this.voxelSize, this.B0, this.B0dir, this.dTE);
            
            fitting                     = [];
            fitting.iteration           = 500;
            fitting.tol                 = 1e-6;
            fitting.initialLearnRate    = 0.01;
            fitting.decayRate           = 0.001;
            fitting.convergenceValue    = 1e-5;
            
            extraData                   = [];
            extraData.weights           = weights;
            
            [~, out] = evalc('objGPU.estimate(f,M,extraData,fitting)');

            chi_out     = out.final.chi_b;
            M_eroded    = imerode(M, strel('sphere',3));                     % avoid noisy boundary phase
            chi_in      = this.TKD(out.final.localField.*M_eroded/this.gyro/this.B0, M_eroded, this.thres_tkd);
            chi_est     = chi_out .* ~M + chi_in .* M;
            
            options = optimoptions('lsqcurvefit', 'Display', 'off');
            % --- Step 2a: inside-M trend vs R2* ---
            mask_good = and(M_eroded, ~bad);
            edges = 0:1:max(R2s(mask_good));                                       % 1-Hz bins
            [chi_med_in, R2_bin_centers] = this.bin_and_median(abs(chi_est), R2s, mask_good, edges);
            sigmoid = @(p,x) (p(2)-p(1)) ./ (1+exp(-(x-p(3))/p(4))) + p(1);  % [sigma1 sigma2 s1 s2]
            p0   = [min(chi_med_in), max(chi_med_in), median(R2_bin_centers), 5];
            lb_s = [0.005, 0.05, 10, 1];
            ub_s = [0.2, 5, 500, 200];
            p_in = lsqcurvefit(sigmoid, p0, R2_bin_centers, chi_med_in, lb_s, ub_s, options);
            % should output verbose here to report coefficients
            
            % --- Step 2b: outside-M trend vs distance ---
            % NOTE: bwdist returns distance in VOXELS, so this scaling is
            % only correct for isotropic voxels. For anisotropic data the
            % distance is wrong by up to the voxel aspect ratio, which
            % distorts the cubic decay fit. Use bwdistsc or a physical
            % distance transform if anisotropic support is needed.
            distMap = bwdist(M, 'euclidean') .* mean(this.voxelSize);            % mm
            edges_d = 0:1:max(distMap(~M),[],'all');
            [chi_med_out, dist_bin_centers] = this.bin_and_median(abs(chi_est), distMap, ~M, edges_d);
            cubic = @(p,x) p(1) ./ (1 + x/p(2)).^3;                          % [sigma0 r0]
            % p0d = [max(chi_med_out), 1];
            p0d = [1 0.1];
            lb_d = [0.01, 0.001];
            ub_d = [5, 20];
            p_out = lsqcurvefit(cubic, p0d, dist_bin_centers, chi_med_out, lb_d, ub_d, options);
            % should output verbose here to report coefficients
            
            % --- Step 3: assemble P, normalized so sigma1 -> 1 ---
            sigma1  = p_in(1);
            P       = zeros(size(f),'like',f);
            P(M)    = sigmoid(p_in, R2s(M)) ./ sigma1;
            P(~M)   = cubic(p_out, distMap(~M)) ./ sigma1;
            P = max(P, 1);   % guard: soft-tissue baseline weight should not fall below ~1
            P = min(P, this.Ps);  % set a cap
        
            fprintf('done.\n')
        end

        function chi = TKD(this,localField,mask,thre_tkd)

            matrixSize = size(localField);

            % dipole kernel
            kernel = this.dipole_kernel(matrixSize,this.voxelSize,this.B0dir);
            
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

        function [ data, mask, extraData ] = prepare_data(this, data, mask, extraData, fitting)

            [r2s,s0] = this.R2star_trapezoidal(data,this.te);

            matrixSize  = size(data,1:3);

            [kernel,~]  = this.dipole_kernel(matrixSize,this.voxelSize,this.B0dir);

            % dimensionless kernel (|D| <= 2/3), so the forward maps ppm -> ppm.
            % Field-strength and TE scaling live in phi2, not here.
            extraData.D = kernel;

            % ----- validate the preconditioner -----
            switch lower(fitting.precond)
                case 'auto'
                    P           = this.compute_adaptive_preconditioner(extraData.totalField, data, mask);
                    extraData.P = P;

                case 'empirical'
                    P           = ones(matrixSize,'single');
                    P(~mask)    = this.Ps;
                    extraData.P = P;

                case 'none'
                    extraData.P = ones(matrixSize,'single');

                case 'emp+r2s'

                    if ~isfield(extraData,'R2s')
                        R2s = this.R2star_trapezoidal(data,this.te);
                    else
                        R2s = extraData.R2s;
                    end

                    % Liu 2017 empirical + R2* preconditioner. The 30 Hz
                    % cutoff is the value implied by the Liu 2020 sigmoid
                    % reducing to this hard threshold (sigma1=1, sigma2=30,
                    % s1=30, s2<<1). Verify against the 2017 paper before
                    % quoting it.
                    %
                    % NOTE: R2s is not sanitised here, unlike in
                    % compute_adaptive_preconditioner. Inf from the
                    % trapezoidal fit passes the > 30 test, so pure-noise
                    % voxels get labelled as strong sources. Apply the same
                    % thres_R2s clamp before thresholding.
                    P                   = ones(matrixSize,'single');
                    P(~mask)            = this.Ps;    % background
                    P(mask & R2s > 30)  = this.Ps;    % ICH / calcification

                    extraData.P         = P;

            end

      
            if ~isequal(size(extraData.P), matrixSize)
                error('gpuTFIR2star:sizeMismatch','extraData.P must be the same size as the field map.');
            end

            % ----- morphology mask defaults to plain TV -----
            if ~isfield(extraData,'MG') || isempty(extraData.MG)
                img             = sqrt(sum(abs(data).^2,4));
                extraData.MG    = this.gradient_mask( img, mask, this.voxelSize, @this.fgrad) .* mask >0;
            end

            % ----- zero-reference region -----
            if fitting.lambdaCSF > 0 && (~isfield(extraData,'M2') || isempty(extraData.M2))
                R2s             = this.R2star_trapezoidal(data,this.te);
                extraData.M2    = this.extract_CSF(R2s, mask, this.voxelSize);
            end

            % ----- weights for data fidelity -----
            % BUG: `img` is only defined inside the MG branch above. If the
            % caller supplies extraData.MG, this line errors on an undefined
            % variable. Should be extraData.img.
            if ~isfield(extraData,'weights') || isempty(extraData.weights)
                weights             = sum(abs(img).^2,4);
                weights             = weights ./max(weights(:));
                extraData.weights   = weights;

            end
            tmp     = data(:,:,:,1);
            scale   = prctile(abs(s0(and(mask, isfinite(s0)))), 99);   % robust; max is noise-sensitive
            data    = data ./ scale;

            extraData.Nvoxel = nnz(mask);
            extraData.Sr     = real(data);
            extraData.Si     = imag(data); 

            % remove img
            % extraData = rmfield(extraData,'img');

            % ----- channel scaling -----
            % chi is O(0.1) ppm, R2* is O(10-100) Hz. Without this the R2*
            % gradients dominate by ~1e3 and chi is effectively
            % unregularised. Normalise by a robust percentile of the
            % GRADIENT magnitude at structure, not of the values, since it
            % is gradients that enter the norm.
            %
            % chi_ref / R2s_ref should be CONVERGED single-channel TV
            % outputs, not raw initial estimates: noise inflates gradient
            % magnitude, so normalising by a noisy map down-weights whichever
            % channel is noisier. The independent-TV baseline run gives you
            % both for free.
            if fitting.lambdaTNV > 0
                img             = sqrt(sum(abs(data).^2,4));
                extraData.MG    = this.gradient_mask( img, mask, this.voxelSize, @this.fgrad) .* mask >0;
                R2s             = this.R2star_trapezoidal(data,this.te);

                objGPU = gpuPDF(this.voxelSize, this.B0, this.B0dir, this.dTE);
            
                fitting_tmp                     = [];
                fitting_tmp.iteration           = 500;
                fitting_tmp.tol                 = 1e-6;
                fitting_tmp.initialLearnRate    = 0.01;
                fitting_tmp.decayRate           = 0.001;
                fitting_tmp.convergenceValue    = 1e-5;
                
                extraData_tmp                   = [];
                extraData_tmp.weights           = extraData.weights;
                extraData_tmp.totalField           = extraData.totalField;
                
                [~, out] = evalc('objGPU.estimate(extraData_tmp.totalField,mask,extraData_tmp,fitting_tmp)');
    
                M_eroded    = imerode(mask, strel('sphere',5));                     % avoid noisy boundary phase
                chi_ref     = this.TKD(out.final.localField.*M_eroded/this.gyro/this.B0, M_eroded, this.thres_tkd);

                gc0 = sqrt(sum(this.fgrad(chi_ref, this.voxelSize).^2, 4));
                gr0 = sqrt(sum(this.fgrad(R2s, this.voxelSize).^2, 4));
    
                sel = mask & any(extraData.MG < 1, 4);      % voxels near structure
                extraData.scaleChi = prctile(gc0(sel), 90);
                extraData.scaleR2s = prctile(gr0(sel), 90);

                % ----- TNV trust weight (reg_tnv only) -----
                % R2* blooming: a compact paramagnetic source elevates R2* over
                % a shell much larger than the source, while chi recovers the
                % source at its true extent. So at strong sources chi has a
                % sharp edge where R2* has a smooth ramp, and the alignment term
                % would penalise the CORRECT solution and push chi edges
                % outward. Down-weight TNV there and fall back to VTV.
                %
                % Smooth ramp rather than a hard threshold so the objective
                % stays differentiable. r0 matches the 30 Hz cutoff used by the
                % 'emp+r2s' preconditioner - same physical logic.
                r0 = 30; dr = 8;                            % Hz
                extraData.wTNV = single(1 ./ (1 + exp((R2s - r0)/dr)));
                extraData.wTNV(~isfinite(extraData.wTNV )) = 0;

                % ----- structural weight (reg_structural only) -----
                % Per-direction, matching MG's layout: chi is allowed an edge
                % along x only where R2* has one along x. Multiplies MG rather
                % than replacing it.
                gr = this.fgrad(R2s ./ extraData.scaleR2s, this.voxelSize);
                sW = prctile(abs(gr(repmat(sel,1,1,1,3))), 95);
                extraData.wStruct = single(exp(-abs(gr) ./ max(sW, eps('single'))));

            end
    
                % convert datatype to single
                data        = single(data);
                mask        = mask > 0;
                extraData   = utils.struct2single(extraData);

        end

    end

    methods(Static)

        %% signal

        function [r2s,s0] = R2star_trapezoidal(img,te)
        % Fast closed-form R2* from the trapezoidal integral of the decay
        % curve. Approximate, but adequate for binning and masking.
        %
        % Produces Inf/NaN wherever the integral is zero (background air,
        % dead channels). Callers must sanitise: see thres_R2s handling in
        % compute_adaptive_preconditioner.

            % disgard phase information
            img = double(abs(img));
            te  = double(te);
            
            % Trapezoidal approximation of integration
            temp = 0;
            for k = 1:size(img,4)-1
                temp = temp + 0.5*(img(:,:,:,k)+img(:,:,:,k+1))*(te(k+1)-te(k));
            end
            
            % very fast estimation
            r2s = (img(:,:,:,1)-img(:,:,:,end)) ./ temp;

            s0 = img(:,:,:,1) .*exp(r2s .* te(1));

        end
        
        function [dipoleKernel,dKComponents] = dipole_kernel(matrixSize,voxelSize,b0dir)
        % Create dipole kernel in k-space with input matrix dimensions
        % and spatial resolution
        %
        % Output
        % ______
        %   dipoleKernel      : Dipole kernel (in k-space), dimensionless,
        %                       real-valued and even, so the adjoint equals
        %                       the forward (conj(D) == D)
        %
        % Kwok-shing Chan @ DCCN
        % Date created: 24 March 2017
        % Date last modified: 27 September 2017

            if nargin<3
                b0dir = [0 0 1];
            end

            [ky,kx,kz] = meshgrid(-matrixSize(2)/2:matrixSize(2)/2-1, ...
                                  -matrixSize(1)/2:matrixSize(1)/2-1, ...
                                  -matrixSize(3)/2:matrixSize(3)/2-1);

            kx = (kx / max(abs(kx(:)))) / voxelSize(1);
            ky = (ky / max(abs(ky(:)))) / voxelSize(2);
            kz = (kz / max(abs(kz(:)))) / voxelSize(3);

            k2 = kx.^2 + ky.^2 + kz.^2;

            dipoleKernel = fftshift( 1/3 - (kx*b0dir(1) + ky*b0dir(2) + kz*b0dir(3)).^2 ./ (k2 + eps) );

            dKComponents.kx = fftshift(kx);
            dKComponents.ky = fftshift(ky);
            dKComponents.kz = fftshift(kz);
            dKComponents.k2 = dKComponents.kx.^2 + dKComponents.ky.^2 + dKComponents.kz.^2;

        end

        function maskCSF = extract_CSF(R2s, mask, voxelSize, flag_erode, thresh_R2s, opts)
        %EXTRACT_CSF  Segment ventricular CSF as a zero-reference region for QSM.
        %
        %   maskCSF = extract_CSF(R2s, Mask, voxel_size)
        %   maskCSF = extract_CSF(R2s, Mask, voxel_size, flag_erode, thresh_R2s)
        %   maskCSF = extract_CSF(..., 'RadiusCentre', 30, ...)
        %
        % Strategy: CSF has low R2*, so threshold R2* and keep the connected
        % components that intersect the largest low-R2* blobs near the centre of the
        % ROI. The centre constraint is what separates ventricles from cortical CSF
        % and from other low-R2* tissue at the periphery.
        %
        % Input
        % -----
        %   R2s         : R2* map [Hz], same size as Mask. May be empty, in which
        %                 case [] is returned.
        %   Mask        : ROI mask (brain). Logical or numeric.
        %   voxel_size  : [dx dy dz] in mm
        %   flag_erode  : erode the ROI with SMV before analysis (default true)
        %   thresh_R2s  : low-R2* threshold [Hz] (default 5)
        %
        % Name-value
        % ----------
        %   RadiusCentre     : radius of the central sphere [mm]   (default 30)
        %   NumCentreRegions : number of seed components to keep    (default 3)
        %   ErodeRadius      : SMV radius for erosion [mm]          (default 10)
        %   Connectivity     : bwconncomp connectivity              (default 6)
        %
        % Output
        % ------
        %   Mask_ROI_CSF : logical, same size as Mask
        %
        % NOTE: non-finite voxels in R2s are excluded, since NaN < thresh and
        % Inf < thresh are both false. This is the desired behaviour but is silent -
        % check nnz(~isfinite(R2s)) if the result looks unexpectedly small.
        % adapted from MEDI toolbox
        
            arguments
                R2s
                mask
                voxelSize               (1,:) double
                flag_erode              (1,1) logical = true
                thresh_R2s              (1,1) double  = 5
                opts.RadiusCentre       (1,1) double = 30
                opts.NumCentreRegions   (1,1) double = 3
                opts.ErodeRadius        (1,1) double = 10
                opts.Connectivity       (1,1) double = 6
            end
        
            if isempty(R2s)
                maskCSF = [];
                return
            end
        
            mask        = mask > 0;
            matrixSize = size(mask);
        
            if ~isequal(size(R2s), matrixSize)
                error('extract_CSF:sizeMismatch', ...
                      'R2s (%s) and Mask (%s) must be the same size.', ...
                      mat2str(size(R2s)), mat2str(matrixSize));
            end
            if numel(voxelSize) < 3
                voxelSize = [voxelSize(:).' ones(1, 3-numel(voxelSize))];
            end
        
            nMask = nnz(mask);
            if nMask == 0
                warning('extract_CSF:emptyMask','Mask is empty; returning empty CSF ROI.');
                maskCSF = false(matrixSize);
                return
            end
        
            % ---- centroid of the ROI, in mm -------------------------------------
            % Computed from 1D marginals rather than full ndgrid arrays: the original
            % allocated three double volumes just to take three weighted means.
            x = (1:matrixSize(1)).' * voxelSize(1);
            y = (1:matrixSize(2))   * voxelSize(2);
            z = reshape((1:matrixSize(3)) * voxelSize(3), 1, 1, []);
        
            cx = sum(x .* sum(mask,[2 3])) / nMask;
            cy = sum(y .* sum(mask,[1 3])) / nMask;
            cz = sum(z .* sum(mask,[1 2])) / nMask;
        
            % ---- central sphere --------------------------------------------------
            % Implicit expansion; compare squared distances to avoid the sqrt.
            Mask_cen = (x-cx).^2 + (y-cy).^2 + (z-cz).^2 <= opts.RadiusCentre^2;
        
            % ---- optional erosion ------------------------------------------------
            % Kept in a separate variable: the original overwrote Mask, so the final
            % restriction silently used the eroded version.
            if flag_erode
                Mask_use = SMV(mask, matrixSize, voxelSize, opts.ErodeRadius) > 0.999;
            else
                Mask_use = mask;
            end
        
            lowR2s = R2s < thresh_R2s;      % non-finite -> false
        
            % ---- seed components near the centre ---------------------------------
            % Restricted to the ROI, unlike the original. For a brain-sized FOV the
            % 30 mm sphere sits well inside the mask so this is a no-op, but it makes
            % the function safe for small FOVs and non-brain applications.
            CC_cen = bwconncomp(lowR2s & Mask_cen & Mask_use, opts.Connectivity);
        
            if CC_cen.NumObjects == 0
                warning('extract_CSF:noSeed', ...
                        ['No low-R2* component found within %g mm of the ROI centroid ' ...
                         '(threshold %g Hz). Returning empty CSF ROI.'], ...
                        opts.RadiusCentre, thresh_R2s);
                maskCSF = false(matrixSize);
                return
            end
        
            % guard against fewer components than requested - the original indexed
            % idxs(1:3) unconditionally and errored when only 1 or 2 were found
            nSeed = min(opts.NumCentreRegions, CC_cen.NumObjects);
        
            numPixels  = cellfun(@numel, CC_cen.PixelIdxList);
            [~, order] = sort(numPixels, 'descend');
        
            seed              = false(matrixSize);
            seed(vertcat(CC_cen.PixelIdxList{order(1:nSeed)})) = true;
        
            % ---- components of the full ROI that touch a seed --------------------
            % labelmatrix replaces the original loop over every component, which was
            % the dominant cost when the threshold produced many small blobs.
            CC = bwconncomp(lowR2s & Mask_use, opts.Connectivity);
            L  = labelmatrix(CC);
        
            keep = unique(L(seed & L > 0));
        
            if isempty(keep)
                warning('extract_CSF:noOverlap', ...
                        'Seed components did not intersect any ROI component; returning empty CSF ROI.');
                maskCSF = false(matrixSize);
                return
            end
        
            maskCSF = ismember(L, keep);   % already confined to Mask_use via L

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
        % NOTE: lambdaTV and lambdaCSF are NOT defaulted here, but are read
        % unconditionally by regulariser(), prepare_data() and
        % postprocess(). Calling this class without setting both errors with
        % "Reference to non-existent field". Add defaults below, or document
        % them as mandatory in estimate().
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

            % data fidelity formulation; see class header
            if ~isfield(fitting,'isnonlinear');     fitting2.isnonlinear    = false; end

            if fitting2.isnonlinear
                warning(['Nonlinear TFI: the total field spans many wraps, so the objective ' ...
                         'has many local minima. Supply extraData.chi0 and validate against ' ...
                         'the linear branch before trusting the result.']);
            end

            if ~isfield(fitting,'precond');     fitting2.precond    = 'auto';           end
            if ~isfield(fitting,'start');       fitting2.start      = 'prior';          end
            if ~isfield(fitting,'TVmode');      fitting2.TVmode     = 'anisotropic';    end

            fitting2.enableComplex      = false;
            fitting2.isMaskedOut        = false;
            fitting2.isOptimiseMemory   = false;

            % disable built-in TV; regularisation is handled by REG
            fitting2.regularisationType = 'none';
            fitting2.lambda             = {0};

        end

        function display_algorithm_info(fitting)
            disp('--------------');
            disp('Fitting option');
            disp('--------------');
            if fitting.isnonlinear
                disp('Non-linear TFI');
            else
                disp('Linear TFI');
            end
            disp('------------------------------------');

        end

        function [med, centers] = bin_and_median(vals, binVar, mask, edges)
            med = nan(numel(edges)-1,1);
            centers = nan(numel(edges)-1,1);
            for i = 1:numel(edges)-1
                sel = mask & binVar>=edges(i) & binVar<edges(i+1);
                if nnz(sel) > 0
                    med(i) = median(vals(sel));
                    centers(i) = mean([edges(i) edges(i+1)]);
                end
            end
            valid = ~isnan(med);
            med = med(valid); centers = centers(valid);
        end

        function y = slice(x, idx, dim, k)
            idx{dim} = k;
            y = x(idx{:});
        end

    end
end