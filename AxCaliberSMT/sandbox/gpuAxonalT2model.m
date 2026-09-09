classdef gpuAxonalT2model < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 29 September 2025
% Date modified: 26 August 2026 (aligned to gacelle v1.1 conventions,
%                see gpuAxCaliberSMT.m/gpuMEAxCaliberSMT.m)
%
% Implements the surface-based axon-radius-dependent T2 relaxation
% model of:
%   Barakovic, M., Pizzolato, M., Tax, C.M.W., Rudrapatna, U., Magon, S.,
%   Dyrby, T.B., Granziera, C., Thiran, J.-P., Jones, D.K.,
%   Canales-Rodriguez, E.J., 2023. Estimating axon radius using
%   diffusion-relaxation MRI: calibrating a surface-based relaxation
%   model with histology. Front. Neurosci. 17, 1209521.
%   https://doi.org/10.3389/fnins.2023.1209521

% TODO: sort b and te and input full DWI

    properties (GetAccess = public, SetAccess = protected)
    % ===== MODEL PARAMETER CONTRACT =====
    % r     : Axon radius [um]
    % S0    : DW signal [au]
    % k2a   : rate of axon radius induced T2 [um/s]
    % R2a   : Intrinsic neurite R2 [1/s]
    % noise : noise
    %
    % modelParams{k} <-> ub(k) <-> lb(k) <-> startPoint(k) <-> step(k)
    % These five arrays MUST stay the same length and index-aligned.
    % Mutate only as a set, via updateProperty() - never assign into a
    % single element from outside the class, or these will desync.
    %
    % 'noise' is solver-conditional (mcmc only) and is kept LAST so
    % that updateProperty() can strip it by name without hardcoding an
    % index. Any future solver-conditional parameter should likewise
    % go last.
        modelParams     = {  'r'; 'S0';'k2a';'R2a';'noise'};
        ub              = [    5;    5;    4;   20;    0.1];
        lb              = [1e-10;    0;    0;    5;   0.01];
        startPoint      = [    1;  0.1;  2.4;    8;  0.005];
        step            = [  0.8;  0.5; 0.29; 1.07;  0.005];
    end

    properties
    % ===== USER-TUNABLE OPTIONS =====
    % Freely settable by users before fitting; no coupling between these.
        epsilon = utils.epsilon;

        % tissue properties (default values from Barakovic et al., 2023,
        % Front. Neurosci. 17, 1209521, https://doi.org/10.3389/fnins.2023.1209521)
        R2c     = 1/126.97e-3;  % axonal T2 constant term [1/s]
        rho2    = 1.16*2;       % axonal surface to volume ratio [um/s]
    end

    properties (GetAccess = public, SetAccess = protected)
    % ===== ACQUISITION PARAMETERS =====
    % Set once in the constructor from user-provided acquisition info.
    % Read-only after construction.
        te;
    end
    
    methods

        % constructuor
        function this = gpuAxonalT2model(te, tissueProperties)
        % Estimation of axon radius from multi-echo T2 decay using a
        % radius-dependent T2 relaxation model
        % smt = gpuAxonalT2model(te, tissueProperties)
        %       output:
        %           - smt: object of a fitting class
        %
        %       input:
        %           - te:       echo time               [s] (if Nt>1)
        %           - tissueProperties
        %               .R2c:   Intrinsic axonal R2 [1/s] (default:1/126.97e-3)
        %               .rho2:  coeffient of 1/r dependence [um/s] (default:1.16*2)
        %
        %  Authors:
        %  Kwok-Shing Chan (kchan2@mgh.harvard.edu)
        %

            % sequence parameters
            % relaxation
            this.te = single(te(:));    % same length as b or scalar (if Nt==1)

            % user defined
            % relaxation
            if isfield(tissueProperties,'R2c');     this.R2c    = single(tissueProperties.R2c);      end
            if isfield(tissueProperties,'rho2');    this.rho2   = single(tissueProperties.rho2);     end

        end

        % Strip solver-/option-conditional entries from the 5 index-
        % aligned model-parameter arrays (modelParams/ub/lb/startPoint/
        % step), based on fitting.solver/isFitR2a/isFitk2a. Must be
        % called before those arrays are used to build fitting.ub/lb
        % (see fit()), so the arrays this class exposes for a given fit
        % always match what was actually requested.
        function this = updateProperty(this, fitting)

            % property change in related to solver
            if ~strcmpi(fitting.solver,'mcmc')
                idx = find(ismember(this.modelParams,'noise'));
                this.modelParams(idx)       = [];
                this.lb(idx)                = [];
                this.ub(idx)                = [];
                this.startPoint(idx)        = [];
                this.step(idx)              = [];
            end

            % whether fitting axonal R2 or not
            if ~fitting.isFitR2a
                idx = find(ismember(this.modelParams,'R2a'));
                this.modelParams(idx)       = [];
                this.lb(idx)                = [];
                this.ub(idx)                = [];
                this.startPoint(idx)        = [];
                this.step(idx)              = [];
            end
            % whether fitting axon-radius T2 rate or not
            if ~fitting.isFitk2a
                idx = find(ismember(this.modelParams,'k2a'));
                this.modelParams(idx)       = [];
                this.lb(idx)                = [];
                this.ub(idx)                = [];
                this.startPoint(idx)        = [];
                this.step(idx)              = [];
            end

        end
    
        % Print a short human-readable summary of the acquisition (TE
        % list) and the fixed tissue parameters (R2c, rho2) to the
        % console; called once at the start of estimate().
        function display_data_model_info(this)

            disp('=====================================');
            disp('Axon radius mapping based on T2 model');
            disp('=====================================');

            disp('----------------')
            disp('Data Information');
            disp('----------------')
            fprintf('TE (ms)                                    : [%s] \n',num2str(this.te.'*1e3,' %.2f'));
            disp('----------------')

            disp('----------------')
            disp('Fixed parameters');
            disp('----------------')
            disp(['Intrinsic intra-axonal T2 (ms)               : ' num2str(1/this.R2c*1e3,'%.2f')]);
            disp(['Surface-to-volume ratio constant (ms)        : ' num2str(this.rho2,'%.2f')]);
            disp('----------------')

        end

        %% higher-level data fitting functions

        % This is a wrapper of the 'fit' function.
        % The main purpose of this function is to handle memory issue and ensure the input data is correct for 'fit'
        function  [out] = estimate(this, dwi, mask, fitting, extraData, pars0)
        % Top-level entry point: validates/normalises the input image,
        % automatically splits it into z-slice segments if it would
        % not fit on the GPU in one go (see utils.find_optimal_segment_3D
        % in the body below), calls fit() per segment, restitches the
        % results, and optionally saves them to disk.
        %
        % Input
        % -----------
        % dwi       : 4D multi-echo image, [x,y,z,te], te order must match
        %             the order of this.te set in the constructor
        % mask      : 3D signal mask, [x,y,z]
        % fitting   : fitting algorithm parameters (see fit() and
        %             check_set_default() for the full list, including
        %             .autoMemManage/.segmentOverlap/.NSegmentUser,
        %             which control the z-slice memory-management
        %             behaviour of this function)
        % extraData : accepted for interface parity with
        %             gpuAxCaliberSMT/gpuMEAxCaliberSMT's estimate(),
        %             but currently UNUSED -- this model has no
        %             gradient-direction/b-value acquisition table to
        %             pass through (its only per-volume acquisition
        %             parameter is TE, already supplied via this.te)
        % pars0     : structure variable of starting points of fitting (optional)
        %
        % Output
        % -----------
        % out       : output structure contains all parameter estimation results
        %
            % if no pars input at all (not even empty) then use prior
            if nargin < 6; pars0        = []; end
            if nargin < 5; extraData    = []; end

            % display basic info
            this.display_data_model_info;

            % get all fitting algorithm parameters
            fitting = this.check_set_default(fitting);

            %%%%%%%%%%%%%%%% Step 1: Validate all input data %%%%%%%%%%%%%%%%
            % compute rotationally invariant signal if needed
            [dwi, scaleFactor] = this.prepare_dwi_data(dwi,mask);

            % mask sure no nan or inf in data
            [dwi,mask] = utils.remove_img_naninf(dwi,mask);

            % convert datatype to single or logical
            dwi     = single(dwi);
            mask    = mask >0;
            if ~isempty(pars0); for km = 1:numel(this.modelParams); pars0.(this.modelParams{km}) = single(pars0.(this.modelParams{km})); end; end

            %%%%%%%%%%%%%%%% End Step 1 %%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%% Step 2: Memory management %%%%%%%%%%%%%%%%

            % --- [Experimental] estimate memory usage using a small batch of data size ---
            % this method tends to be more conservative than the actual memory ussage.
            % Splits the volume into z-slice segments when the whole
            % dataset would not fit on the GPU at once (e.g. a smaller/
            % older GPU), so estimate() also works there -- transparent
            % to the caller: NSegment == 1 (no splitting) whenever the
            % data already fits, identical to the previous behaviour.
            [seg,NSegment] = utils.find_optimal_segment_3D(this, dwi, mask, fitting, pars0);

            % parameter estimation
            out = [];
            for kseg = 1:NSegment

                if NSegment > 1
                    fprintf('Running #Segment = %d/%d \n',kseg,NSegment);
                    disp   ('------------------------')
                end

                % divide the data; fitRange includes halo slices (if any), ownedRange
                % is what this segment is responsible for writing back
                fitRange                        = seg(kseg).fit;
                ownedRange                      = seg(kseg).owned;
                [dwiSeg, maskSeg, pars0Seg]     = this.slice_segment(dwi, mask, fitRange, pars0);

                % run fitting
                [outSeg] = this.fit(dwiSeg,maskSeg,fitting,pars0Seg);

                % discard halo slices from this segment's output before restoring,
                % so segment boundaries never keep voxels from a neighbour's
                % independently-converged fit (no-op when seg(kseg).fit == .owned)
                outSeg = utils.crop_segment_output(outSeg, seg(kseg));

                % restore 'out' structure from segment
                out = utils.restore_segment_structure(out,outSeg,ownedRange,kseg);

            end

            out.mask = mask;
            % rescale S0 in every result-metric substructure present.
            % askadam produces 'final'/'min'; mcmc produces whichever of
            % 'mean'/'median'/'std'/'iqr'/'mode' fitting.metric requested
            % (std/iqr are scale-equivariant, so rescaling them by the
            % same factor as the point estimates is still correct).
            metric_fields = intersect(fieldnames(out), {'final','min','mean','median','std','iqr','mode'});
            for km = 1:numel(metric_fields)
                fname = metric_fields{km};
                if isfield(out.(fname),'S0')
                    out.(fname).S0 = out.(fname).S0 * scaleFactor;
                end
            end
            %%%%%%%%%%%%%%%% End Step 2 %%%%%%%%%%%%%%%%

            % save the estimation results if the output filename is provided
            switch fitting.solver
                case 'askadam'
                    askadam.save_askadam_output(fitting.outputFilename,out)
                case 'mcmc'
                    mcmc.save_mcmc_output(fitting.outputFilename,out)
            end

        end

        % Data fitting function
        % This is a wapper of the askadam class 'fit' function
        function [out] = fit(this,dwi,mask,fitting, pars0)
        %
        % Input
        % -----------
        % dwi       : S0-normalised (see prepare_dwi_data) 4D multi-echo
        %             image, [x,y,slice,te]; the te order must match
        %             this.te set in the constructor
        % mask      : 3D signal mask, [x,y,slice]
        % fitting   : fitting algorithm parameters
        %   .Niteration         : no. of maximum iterations, default = 10000
        %   .initialLearnRate   : initial gradient step size, defaulr = 0.01
        %   .decayRate          : decay rate of gradient step size; learningRate = initialLearnRate / (1+decayRate*epoch), default = 0.0005
        %   .convergenceValue   : convergence tolerance, based on the slope of last 'convergenceWindow' data points on loss, default = 1e-8
        %   .convergenceWindow  : number of data points to check convergence, default = 20
        %   .tol                : stop criteria on metric value, default = 1e-3
        %   .lambda             : regularisation parameter, default = 0 (no regularisation)
        %   .TVmode             : mode for TV regulariation, '2D'|'3D', default = '2D'
        %   .regmap             : parameter map used for regularisation, default = {'r'}
        %   .solver             : 'askadam'|'mcmc', default = 'askadam'
        %   .isFitR2a           : whether to fit R2a (otherwise fixed at this.R2c), default = false
        %   .isFitk2a           : whether to fit k2a (otherwise fixed at this.rho2), default = false
        %   .lossFunction       : loss for data fidelity term, 'L1'|'L2'|'MSE', default = 'L1'
        %   .display            : online display the fitting process on figure, true|false, defualt = false
        % pars0     : structure variable of starting points of fitting (optional)
        %
        % Output
        % -----------
        % out       : output structure
        %   .final      : final results (askadam only)
        %   .min        : results with the minimum loss metric across all iterations (askadam only)
        %       .loss       : loss metric
        %   .mean/.median/.std/.iqr/.mode : posterior summary statistics
        %       requested via fitting.metric (mcmc only)
        %
        % Kwok-Shing Chan @ MGH
        % kchan2@mgh.harvard.edu
        % Date created: 1 Oct 2025
        % Date modified: 26 August 2026 (added mcmc solver support)
        %
            
            
            % check image size
            dims = size(dwi,1:3);

            %%%%%%%%%%%%%%%%%%%% Step 1. Validate and parse input %%%%%%%%%%%%%%%%%%%%
            if nargin < 3 || isempty(mask); mask = ones(dims,'logical'); end % if no mask input then fit everthing
            if nargin < 4; fitting = struct(); end
            % set initial tarting points
            if nargin < 5; pars0 = []; % no initial starting points
            else
                if ~isempty(pars0); for km = 1:numel(this.modelParams); pars0.(this.modelParams{km}) = single(pars0.(this.modelParams{km})); end; end
            end

            % get all fitting algorithm parameters 
            fitting                 = this.check_set_default(fitting);
            % determine fitting parameters
            this                    = this.updateProperty(fitting);
            fitting.modelParams     = this.modelParams;
            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(fitting.modelParams)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(fitting.modelParams)); end

            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run optimisation and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w       = this.compute_optimisation_weights(mask); % This is a customised funtion

            % 2.2 estimate prior if needed
            if isempty(pars0);  pars0 = this.determine_x0(dwi,mask,fitting); end

            % 2.3 optimisation main
            switch fitting.solver
                case 'askadam'
                    out         = askadam().optimisation(dwi, mask, w, pars0, fitting, @this.FWD);
                case 'mcmc'
                    fitting.xStepSize = this.step;

                    out         = mcmc().optimisation(dwi, mask, w, pars0, fitting, @this.FWD);
            end

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            disp('The estimation is completed.');
            
            % clear GPU
            reset(gpuDevice)

        end

        %% Data preparation

        % Normalise the input image to an O(1) intensity scale before
        % fitting (unlike gpuAxCaliberSMT/gpuMEAxCaliberSMT, this model
        % has no gradient directions to reduce to a rotationally
        % invariant signal -- 'dwi' here is already a plain multi-echo
        % magnitude image). A quick log-linear T2 fit (R2_lsq) gives an
        % approximate per-voxel S0 map; its 95th percentile across the
        % mask is used as a single global scale factor (robust to a few
        % anomalously bright voxels) so the fitted S0 stays within its
        % [lb,ub] bounds regardless of the input image's raw units.
        % scaleFactor is later used in estimate() to rescale S0 back to
        % the input's original units.
        function [dwi, scaleFactor] = prepare_dwi_data(this,dwi,mask)

            [~,m0] = this.R2_lsq(dwi,this.te,mask);

            scaleFactor = prctile(m0(mask>0),95);

            dwi = dwi./ scaleFactor;

        end

        % segment data based on slice (used by estimate()'s
        % find_optimal_segment_3D memory-management loop)
        function [dwiSeg, maskSeg, pars0Seg] = slice_segment(this, dwi, mask, slice, pars0)

            dwiSeg      = dwi(:,:,slice,:,:,:,:,:,:);
            maskSeg     = mask(:,:,slice);
            if ~isempty(pars0)
                for km = 1:numel(this.modelParams)
                    pars0Seg.(this.modelParams{km}) = pars0.(this.modelParams{km})(:,:,slice);
                end
            else
                pars0Seg = [];
            end

        end

        % compute weights for optimisation
        function w = compute_optimisation_weights(this,mask)
        %
        % Output
        % ------
        % w         : N-D signal masked wegiths
        %
        % Unlike gpuAxCaliberSMT/gpuMEAxCaliberSMT (which weight each
        % b-shell/SH-order differently via a lmax-dependent scheme,
        % since their data are rotationally-invariant Sl coefficients),
        % this model's data are plain per-TE signal, so every TE volume
        % gets an equal (mask-only) weight.
            w = single(repmat(mask,1,1,1,numel(this.te)));

        end

        %%%%% Prior estimation related functions %%%%%

        % Build the per-voxel starting-point structure x0 for the
        % optimiser, according to fitting.start:
        %   'likelihood' : closed-form log-linear T2 fit (estimate_prior)
        %   'default'    : same fixed startPoint value for every voxel
        %   (numeric)    : user-supplied fixed value(s) for every voxel,
        %                  same order/length as this.modelParams
        % In the 'default' and numeric-vector cases, R2a/k2a (when not
        % being fitted) are afterwards overwritten with their fixed
        % class constants (this.R2c/this.rho2), since they are global
        % constants shared by every voxel, not per-voxel unknowns.
        %
        % Input
        % -----------
        % y         : S0-normalised 4D multi-echo image, [x,y,z,te]
        % mask      : 3D signal mask, [x,y,z]
        % fitting   : fitting algorithm parameters (see fit())
        %
        % Output
        % -----------
        % x0        : structure of per-voxel starting points, one field
        %             per entry of this.modelParams
        function x0 = determine_x0(this,y,mask,fitting)

            disp('---------------');
            disp('Starting points');
            disp('---------------');

            dims = size(mask,1:3);

            if ischar(fitting.start)
                switch lower(fitting.start)
                    case 'likelihood'
                        % using maximum likelihood method to estimate starting points
                        x0 = this.estimate_prior(y,mask);
                        % R2a and k2a are global constants
                        % if any(ismember(this.modelParams,'R2a')); x0.R2a = this.R2c;    end
                        % if any(ismember(this.modelParams,'k2a')); x0.k2a = this.rho2;   end
    
                    case 'default'
                        % use fixed points
                        fprintf('Using default starting points for all voxels at [%s]: [%s]\n', cell2str(this.modelParams),replace(num2str(this.startPoint(:).',' %.2f'),' ',','));
                        x0 = utils.initialise_x0(dims,this.modelParams,this.startPoint);
                        % R2a and k2a are global constants
                        if any(ismember(this.modelParams,'R2a')); x0.R2a = this.R2c;    end
                        if any(ismember(this.modelParams,'k2a')); x0.k2a = this.rho2;   end

                end
            else
                % user defined starting point
                x0 = fitting.start(:);
                fprintf('Using user-defined starting points for all voxels at [%s]: [%s]\n',cell2str(this.modelParams),replace(num2str(x0(:).',' %.2f'),' ',','));
                x0 = utils.initialise_x0(dims,this.modelParams,x0);
                if any(ismember(this.modelParams,'R2a')); x0.R2a = this.R2c;    end
                if any(ismember(this.modelParams,'k2a')); x0.k2a = this.rho2;   end

            end

            % make sure the input is bounded
            x0 = askadam.set_boundary(x0,fitting.ub,fitting.lb);
            
            fprintf('Estimation lower bound [%s]: [%s]\n',      cell2str(this.modelParams),replace(num2str(fitting.lb(:).',' %.2f'),' ',','));
            fprintf('Estimation upper bound [%s]: [%s]\n',      cell2str(this.modelParams),replace(num2str(fitting.ub(:).',' %.2f'),'  ',','));
            disp('---------------');
        end

        % using maximum likelihood method to estimate starting points
        function pars0 = estimate_prior(this,dwi,mask)
        % Closed-form starting-point estimate for gpuAxonalT2model:
        % fit a plain log-linear mono-exponential T2 decay per voxel
        % (R2_lsq) to get an observed R2, then invert the forward model
        % R2 = R2c + rho2/r (i.e. r = rho2/(R2-R2c), the model's
        % surface-relaxivity relation from Barakovic et al., 2023) to
        % get a starting radius r0 -- this ignores the k2a scaling that
        % gets fitted separately, i.e. it only holds exactly when
        % k2a == rho2 (the class default), but is a reasonable estimate
        % for starting the nonlinear optimiser regardless.

            disp('Estimate starting points based on likelihood ...')

            [r2,s0]  = this.R2_lsq(dwi,this.te,double(mask));

            % convert to radius
            r_max       = 20; % um, upper bound
            r           = this.rho2./(r2 - this.R2c); % inverse of R2 = R2c + rho2/r
            r(r<0)      = 0;
            r(r>r_max)  = r_max;
            r           = r .* mask;

            s0(isnan(s0))       = this.lb(2);
            s0(isinf(s0))       = this.lb(2);
            s0(s0<0)            = 0;
            s0(s0>this.ub(2))   = this.ub(2);

            %  initiate pars0 with the same order as modelParams
            for k = 1:numel(this.modelParams)
                pars0.(this.modelParams{k}) = [];
            end

            pars0.r     = r;
            pars0.S0    = s0;
            % global constant for the data
            if any(ismember(this.modelParams,'k2a'))
                pars0.k2a  = this.rho2;
            end
            if any(ismember(this.modelParams,'R2a'))
                pars0.R2a  = this.R2c;
            end
            % noise (mcmc only): the likelihood method above has no
            % closed-form noise estimate, so seed it uniformly at its
            % class startPoint, same as gpuAxCaliberSMT/gpuMEAxCaliberSMT
            if strcmpi(this.modelParams{end},'noise')
                pars0.(this.modelParams{end}) = single(ones(size(mask)) * this.startPoint(end));
            end

        end

        %%  Signal related functions

        % Forward model: predicts the (S0-normalised) multi-echo signal
        % for a given set of parameters. R2a/k2a fall back to their
        % fixed class constants (this.R2c/this.rho2) when not present
        % as fields of pars, i.e. when fitting.isFitR2a/isFitk2a are
        % false and they are therefore not being estimated per voxel.
        % Called by askadam()/mcmc().optimisation() as the model
        % function handle (@this.FWD); works unmodified for both
        % solvers since signal_axonal_T2 is a simple, smooth elementwise
        % expression (no arrayfun-based branch needed, unlike the
        % diffusion-restriction terms in gpuAxCaliberSMT/gpuMEAxCaliberSMT).
        function s = FWD(this, pars)

            % minimal fitting parameters
            r   = pars.r;
            S0  = pars.S0;
            if isfield(pars,'R2a')
                R2a = pars.R2a;
            else
                R2a = this.R2c;
            end
            if isfield(pars,'k2a')
                k2a = pars.k2a;
            else
                k2a = this.rho2;
            end

            % intra-axonal signal
            s = this.signal_axonal_T2(this.te,S0,r, R2a, k2a) ;
                
        end

    end

    methods(Static)

        %% Utility
        %%%%%%%%%% Compartmental signal
        % Surface-based axon-radius-dependent T2 relaxation model:
        % R2(r) = R2a + k2a/r, i.e. intrinsic intra-axonal R2 plus a
        % surface-relaxivity term that grows as the axon's
        % surface-to-volume ratio increases (smaller r). See
        % Barakovic et al., 2023, Front. Neurosci. 17, 1209521,
        % https://doi.org/10.3389/fnins.2023.1209521
        function S = signal_axonal_T2(te,s0,r, R2a, k2a)

            S = s0.* exp(-te.*( R2a + k2a./r));

        end

        % Voxelwise closed-form mono-exponential T2 fit: log(|img|) is
        % linear in te with slope -R2 and intercept log(S0), so a
        % per-voxel linear least-squares fit (x\y) over all TEs at once
        % recovers R2 and S0 directly, without any nonlinear solver.
        % Used both to build a scale factor in prepare_dwi_data and as
        % the starting-point method in estimate_prior. R2 is clamped to
        % [1/(20*max(te)), 20/min(te)] to keep pathological fits (e.g.
        % from near-zero or noise-dominated signal) within a plausible
        % T2 range.
        %
        % Input
        % -----------
        % img       : 4D multi-echo magnitude image, [x,y,z,te]
        % te        : 1D echo times, same length as img's 4th dimension
        % mask      : 3D signal mask, [x,y,z]
        %
        % Output
        % -----------
        % r2        : 3D R2 map [1/te's time unit], masked
        % m0        : 3D extrapolated S0 (signal at te=0) map, masked
        function [r2,m0] = R2_lsq(img,te,mask)

            img = double(img);
            te  = double(te);

            % set range of R2 and T2
            minT2s      = min(te)/20;
            maxT2s      = max(te)*20;
            ranger2     = [1/maxT2s, 1/minT2s];
            
            [nx,ny,nz,nt] = size(img);
            
            x       = ones(nt,2);
            x(:,2)  = -te(:);
            y       = permute(log(abs(img)),[4 1 2 3]);
            
            y       = reshape(y,[size(y,1) numel(y)/size(y,1)]);
            b       = x\y;
            r2      = reshape( b(2,:),nx,ny,nz) .* mask;
            m0      = exp(reshape( b(1,:),nx,ny,nz)) .* mask;
            
            r2(r2>max(ranger2)) = max(ranger2);
            r2(r2<min(ranger2)) = min(ranger2);

            m0(m0<0) = 0;

        end

        %%%%%%%%%%%%%%
        % Fill in any missing field of the 'fitting' options struct
        % with its default value. Delegates the generic solver options
        % to askadam.check_set_default_basic/mcmc.check_set_default_basic
        % (chosen by fitting.solver, default 'askadam'), then adds this
        % model's own options (regmap/start/isFitR2a/isFitk2a).
        function fitting2 = check_set_default(fitting)
            if ~isfield(fitting,'solver');      fitting.solver = 'askadam';        end

            % get basic fitting setting check
            if strcmpi(fitting.solver,'mcmc')

                % mcmc
                fitting2                = mcmc.check_set_default_basic(fitting);
                fitting2.lossFunction   = 'l2'; % for computing weights

            else

                % askadam
                fitting2 = askadam.check_set_default_basic(fitting);

            end

            % get customised fitting setting check
            if ~isfield(fitting,'regmap');      fitting2.regmap     = {'r'};            end
            if ~isfield(fitting,'start');       fitting2.start      = 'likelihood';     end
            if ~isfield(fitting,'isFitR2a');    fitting2.isFitR2a   = false;            end
            if ~isfield(fitting,'isFitk2a');    fitting2.isFitk2a   = false;            end

            if ~iscell(fitting2.regmap)
                fitting2.regmap = cellstr(fitting2.regmap);
            end

        end
    
    end

end