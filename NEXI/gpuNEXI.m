classdef gpuNEXI < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 8 Dec 2023 (v0.1.0)
% Date modified: 29 March 2024 (v0.2.0)
% Date modified: 4 April 2024 (v0.3.0)
% Date modified: 20 August 2024 (v0.4.0)
% Date modified: 8 June 2026 (v0.5.0) (update memory management, support mcmc in the same function)

    properties (GetAccess = public, SetAccess = protected)
    % ===== MODEL PARAMETER CONTRACT =====
    % fa        : Neurite volume fraction
    % Da        : longitudinal diffusivity of neurite [ms/us^2]
    % De        : diffusivity of extracellular water [ms/us^2]
    % ra        : exchange rate from neurite to extracellular space [1/s]
    % p2        : non-linear neurite dispersion index
    % noise     : noise
    %
    % modelParams{k} <-> ub(k) <-> lb(k) <-> startPoint(k) <-> step(k)
    % These five arrays MUST stay the same length and index-aligned.
    % Mutate only as a set, via updateProperty() - never assign into a
    % single element from outside the class, or these will desync.
    %
    % 'noise' is solver-conditional (mcmc only) and is kept LAST so that
    % updateProperty() can strip it by name without hardcoding an index,
    % and fit() (~line 521) can locate it via this.modelParams{end}. Any
    % future solver-conditional parameter should likewise go last.
        modelParams     = {'fa'  ;'Da'  ;'De'   ;'ra'   ;'p2'; 'noise'};
        ub              = [   1  ;   3  ;   3   ;   1   ;  1 ;     0.1];
        lb              = [ eps  ; eps  ; eps   ;1/250  ; eps;    0.01];
        startPoint      = [ 0.4  ;   2  ;   1   ; 0.05  ; 0.2;   0.005];
        step            = [  0.05;  0.15;   0.15;  0.005;0.05;   0.005];
    end

    properties
    % ===== USER-TUNABLE OPTIONS =====
    % Freely settable by users before fitting; no coupling between these.
        thres_similarity    = 0.1; 
        thres_impossible    = 0.1;
        thres_bkg           = 0.01;

        seed = 48463;   % for reproducible random number generation

        epsilon = utils.epsilon;

    end

    properties (GetAccess = public, SetAccess = protected)
    % ===== ACQUISITION PARAMETERS =====
    % Set once in the constructor from user-provided acquisition info.
    % Read-only after construction.
        b;
        Delta;  
        Nav;
    end

    properties (Access = protected)
    % ===== SURFACE GEOMETRY CACHE =====
    % Populated by get_surface_geometry(); recomputed only when (surf_dir,
    % depth, hemisphere) differ from the previous call on this object, since
    % utils.get_surface_neighbours is not cheap to recompute (unvectorised
    % per-face adjacency loop over the full mesh).
        % surfaceGeometryCache = struct('surf_dir','','depth',[],'hemisphere',{{}},'neighbours',[],'dr',[],'edgeLength',[]);
        surfaceGeometryCache = struct('surf_dir','','depth',[],'hemisphere',{{}},'neighbours',[],'edgeLength',[]);
    end

    methods

        % constructuor
        function this = gpuNEXI(b, Delta, varargin)
        % NEXI Exchange rate estimation using NEXI model
        % obj = gpuNEXI(b, Delta, Nav)
        %
        % Input
        % ----------
        % b         : b-value [ms/um2]
        % Delta     : gradient seperation [ms]
        % Nav       : # gradient direction for each b-shell (optional)
        %
        % Output
        % ----------
        % obj       : object of a fitting class
        %
        % Author:
        %  Kwok-Shing Chan (kchan2@mgh.harvard.edu) 
        %  Hong-Hsi Lee (hlee84@mgh.harvard.edu)
        %  Copyright (c) 2023 Massachusetts General Hospital
        %
        %  Adapted from the code of
        %  Dmitry Novikov (dmitry.novikov@nyulangone.org)
        %  Copyright (c) 2023 New York University
            
            % handle full b-value vector and Big delta vector
            [bval_sorted,~,BDELTA_sorted,~] = DWIutility.unique_shell_keepb0(b,[],Delta,[],true);

            this.b      = single(bval_sorted(:)) ;
            this.Delta  = single(BDELTA_sorted(:)) ;
            if nargin > 2
                this.Nav = varargin{1} ;
            else
                this.Nav = ones(size(this.b)) ;
            end
            this.Nav = this.Nav(:) ;
        end
        
        % update properties according to lmax
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

            % lmax = 0
            if fitting.lmax == 0
                idx = find(ismember(this.modelParams,'p2'));
                this.modelParams(idx)     = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startPoint(idx)      = [];
                this.step(idx)            = [];
            end

        end

        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('=============================================');
            disp('Neurite Exchange Imaging (NEXI): Narrow pulse');
            disp('=============================================');

            disp('----------------')
            disp('Data Information');
            disp('----------------')
            fprintf('b-shells (ms/um2)              : [%s] \n',num2str(this.b.',' %.2f'));
            fprintf('Diffusion time (ms)            : [%s] \n',num2str(this.Delta.',' %i'));
            disp('----------------');
        end

        %% higher-level data fitting functions
        % Wrapper function of fit to handle image data; automatically segment data and fitting in case the data cannot fit in the GPU in one go
        function  [out] = estimate(this, data, mask, extradata, fitting, pars0)
        % Perform NEXI model parameter estimation based on askAdam
        % Input data are expected in multi-dimensional image
        % 
        % Input
        % -----------
        % data      : 4D DWI, [x,y,z,dwi]
        % mask      : 3D signal mask, [x,y,z]
        % extradata : Optional additional data
        %   .bval       : 1D bval in ms/um2, [1,dwi]                (Optional, only needed if dwi is full acquisition)
        %   .bvec       : 2D b-table, [3,dwi]                       (Optional, only needed if dwi is full acquisition)
        %   .ldelta     : 1D gradient pulse duration in ms, [1,dwi] (Optional, only needed if dwi is full acquisition)
        %   .BDELTA     : 1D diffusion time in ms, [1,dwi]          (Optional, only needed if dwi is full acquisition)
        %   .sigma      : 3D noise map, [x,y,z]                     (Optional, only needed for NEXIrice model)
        %   .surf_dir   : FreeSurfer surf directory                 (Required if fitting.dataType='surface')
        %   .hemisphere : cell array subset of {'lh','rh'}          (Required if fitting.dataType='surface'; must match mask dim3/Nhemi)
        %   .depth      : cortical depth, [0,1]                     (Optional, surface mode only, default 0.5)
        % fitting   : fitting algorithm parameters (see fit function)
        %   .dataType   : 'volumetric' (default) | 'surface'. Surface mode expects
        %                 data/mask in [1, Nvertex, Nhemi] convention and ignores
        %                 fitting.TVmode entirely (regularisation dispatched via
        %                 surface_total_variation instead of the built-in volumetric
        %                 TV operator). Segmentation for surface mode is opt-in and
        %                 exact-only (fitting.NSegmentUser must be 1 or Nhemi).
        % pars0     : (Optional) initial starting points for model parameters
        % 
        % Output
        % -----------
        % out       : output structure contains all estimation results
        % fa        : Intraneurite volume fraction
        % Da        : Intraneurite diffusivity (um2/ms)
        % De        : Extraneurite diffusivity (um2/ms)
        % ra        : exchange rate from intra- to extra-neurite compartment
        % p2        : dispersion index (if fitting.lax=2)
        % 
            
            % display basic info
            this.display_data_model_info;

            % get all fitting algorithm parameters 
            fitting     = this.check_set_default(fitting);

            %%%%%%%%%%%%%%%% Step 1: Validate all input data %%%%%%%%%%%%%%%%
            % compute rotationally invariant signal if needed
            % [data,mask] = this.prepare_dwi_data(data,mask,extradata,fitting.lmax);
            [data, mask, extradata] = this.prepare_dwi_data(data,mask,extradata,fitting.lmax,fitting);

            % if no pars input at all (not even empty) then use prior
            if nargin < 6; pars0 = []; end

            % convert datatype to single
            data    = single(data);
            mask    = mask >0;
            if ~isempty(pars0); for km = 1:numel(this.modelParams); pars0.(this.modelParams{km}) = single(pars0.(this.modelParams{km})); end; end

            %%%%%%%%%%%%%%%% End Step 1 %%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%% Step 2: Memory management %%%%%%%%%%%%%%%%

            % --- [Experimental] estimate memory usage using a small batch of data size ---
            % this method tends to be more conservative than the actual memory ussage
            [seg,NSegment] = utils.find_optimal_segment_3D(this, data, mask, fitting, pars0);

            % parameter estimation
            out = [];
            for kseg = 1:NSegment

                if NSegment > 1
                    fprintf('Running #Segment = %d/%d \n',kseg,NSegment);
                    disp   ('------------------------')
                end

                % divide the data; fitRange includes halo slices (if any), ownedRange
                % is what this segment is responsible for writing back
                fitRange                                    = seg(kseg).fit;
                ownedRange                                  = seg(kseg).owned;
                [dataSeg, maskSeg, pars0Seg, extradataSeg]  = this.slice_segment(data, mask, fitRange, pars0, extradata);

                % run fitting
                [outSeg] = this.fit(dataSeg,maskSeg,fitting,pars0Seg,extradataSeg);

                % discard halo slices from this segment's output before restoring,
                % so segment boundaries never keep voxels from a neighbour's
                % independently-converged fit (no-op when seg(kseg).fit == .owned)
                outSeg = utils.crop_segment_output(outSeg, seg(kseg));

                % restore 'out' structure from segment
                out = utils.restore_segment_structure(out,outSeg,ownedRange,kseg);

            end
            out.mask = mask;

            %%%%%%%%%%%%%%%% End Step 2 %%%%%%%%%%%%%%%%

            % save the estimation results if the output filename is provided
            % askadam.save_askadam_output(fitting.outputFilename,out)
            switch fitting.solver
                case 'askadam'
                    askadam.save_askadam_output(fitting.outputFilename,out)
                case 'mcmc'
                    mcmc.save_mcmc_output(fitting.outputFilename,out)
            end


        end

        % Cached wrapper around utils.get_surface_neighbours; recomputes only
        % when (surf_dir, depth, hemisphere) differ from the last call on
        % this object, so repeated estimate() calls on the same subject/
        % surface don't repay the mesh-adjacency computation each time.
        function [neighbours, edgeLength] = get_surface_geometry(this, surf_dir, depth, hemisphere)

            if nargin < 3 || isempty(depth);      depth      = 0.5;         end
            if nargin < 4 || isempty(hemisphere); hemisphere = {'lh','rh'}; end
            if ~iscell(hemisphere); hemisphere = cellstr(hemisphere); end

            cacheHit = strcmp(this.surfaceGeometryCache.surf_dir, surf_dir) && ...
                       isequal(this.surfaceGeometryCache.depth, depth) && ...
                       isequal(this.surfaceGeometryCache.hemisphere, hemisphere);

            if cacheHit
                neighbours = this.surfaceGeometryCache.neighbours;
                % dr         = this.surfaceGeometryCache.dr;
                edgeLength = this.surfaceGeometryCache.edgeLength;
                return
            end

            [neighbours, ~, edgeLength] = utils.get_surface_neighbours(surf_dir, depth, hemisphere);
            neighbours = (single(neighbours));
            % dr         = (single(dr));
            edgeLength = (single(edgeLength));

            this.surfaceGeometryCache.surf_dir   = surf_dir;
            this.surfaceGeometryCache.depth      = depth;
            this.surfaceGeometryCache.hemisphere = hemisphere;
            this.surfaceGeometryCache.neighbours = neighbours;
            % this.surfaceGeometryCache.dr         = dr;
            this.surfaceGeometryCache.edgeLength = edgeLength;

        end

        % Data fitting function, can be 2D (voxel-based) or 4D (image-based)
        function [out] = fit(this,dwi,mask,fitting,pars0,extradata)
        %
        % Input
        % -----------
        % dwi       : S0 normalised 4D dwi images, [x,y,slice,diffusion], 4th dimension corresponding to [Sl0_b1,Sl0_b2,Sl2_b1,Sl2_b2, etc.]; the order of bval must match the order in the constructor gpuNEXI
        % mask      : 3D signal mask, [x,y,slice]
        % fitting   : fitting algorithm parameters
        %   .Nepoch             : no. of maximum iterations, default = 4000
        %   .initialLearnRate   : initial gradient step size, defaulr = 0.01
        %   .decayRate          : decay rate of gradient step size; learningRate = initialLearnRate / (1+decayRate*epoch), default = 0.0005
        %   .convergenceValue   : convergence tolerance, based on the slope of last 'convergenceWindow' data points on loss, default = 1e-8
        %   .convergenceWindow  : number of data points to check convergence, default = 20
        %   .tol                : stop criteria on metric value, default = 1e-3
        %   .lambda             : regularisation parameter, default = 0 (no regularisation)
        %   .TVmode             : mode for TV regulariation, '2D'|'3D', default = '2D'
        %   .regmap             : parameter map used for regularisation, 'fa'|'ra'|'Da'|'De', default = 'fa'
        %   .lmax               : Order of rotational invariant, 0|2, default = 0
        %   .lossFunction       : loss for data fidelity term, 'L1'|'L2'|'MSE', default = 'L1'
        %   .display            : online display the fitting process on figure, true|false, defualt = false
        % pars0     : 4D parameter starting points of fitting, [x,y,slice,param], 4th dimension corresponding to fitting  parameters with order [fa,Da,De,ra,p2] (optional)
        % 
        % Output
        % -----------
        % out       : output structure
        %   .final      : final results
        %       .fa         : Intraneurite volume fraction
        %       .Da         : Intraneurite diffusivity (um2/ms)
        %       .De         : Extraneurite diffusivity (um2/ms)
        %       .ra         : exchange rate from intra- to extra-neurite compartment
        %       .p2         : dispersion index (if fitting.lax=2)
        %       .loss       : final loss metric
        %   .min        : results with the minimum loss metric across all iterations
        %       .fa         : Intraneurite volume fraction
        %       .Da         : Intraneurite diffusivity (um2/ms)
        %       .De         : Extraneurite diffusivity (um2/ms)
        %       .ra         : exchange rate from intra- to extra-neurite compartment
        %       .p2         : dispersion index (if fitting.lax=2)
        %       .loss       : loss metric      
        %
        % Description: askAdam Image-based NEXI model fitting
        %
        % Kwok-Shing Chan @ MGH
        % kchan2@mgh.harvard.edu
        % Date created: 8 Dec 2023
        % Date modified: 3 April 2024
        %
        %
            
            % get image size
            dims = size(dwi,1:3);

            %%%%%%%%%%%%%%%%%%%% 1. Validate and parse input %%%%%%%%%%%%%%%%%%%%
            if nargin < 3 || isempty(mask); mask = ones(dims,'logical'); end % if no mask input then fit everthing
            if nargin < 4; fitting = struct(); end
            % set initial tarting points
            if nargin < 5; pars0 = []; % no initial starting points
            else
                if ~isempty(pars0); for km = 1:numel(this.modelParams); pars0.(this.modelParams{km}) = single(pars0.(this.modelParams{km})); end; end
            end
            if nargin < 6; extradata = []; end % only used by the surface TV path below

            % get all fitting algorithm parameters 
            fitting                 = this.check_set_default(fitting);
            % determine fitting parameters
            this                    = this.updateProperty(fitting);
            fitting.modelParams     = this.modelParams;
            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(fitting.modelParams)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(fitting.modelParams)); end

            % gacelle does not check your extra data, so put them onto gpu now
            if ~isempty(extradata); extradata = utils.struct2gpusingle(extradata); end
            
            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w = this.compute_optimisation_weights(mask,fitting.lossFunction,fitting.lmax); % This is a customised funtion

            % 2.2 estimate prior if needed
            if isempty(pars0);  pars0 = this.determine_x0(dwi,mask,fitting); end

            % You may add more dispay messages here
            disp('---------------------------');
            disp('Additional model parameters');
            disp('---------------------------');
            disp(['NEXI rotational invariant model, lmax = ' num2str(fitting.lmax)]);
            disp('---------------------------');

            % 2.3 askAdam optimisation main
            switch fitting.solver
                case 'askadam'
                    if strcmpi(fitting.regularisationType,'prior') && fitting.lambda{1} > 0
                        % Prior-distribution regularisation: custom-regulariser path via
                        % the userFcn/userInput two-slot form, applicable to EITHER
                        % dataType ('volumetric' or 'surface') since prior_distribution_normal
                        % is geometry-agnostic (it goes through utils.reshape_GD2ND/mask
                        % exactly like the forward model does).
                        if isempty(extradata) || ~isfield(extradata,'mu') || ~isfield(extradata,'sigma')
                            error('GACELLE:missingPriorDistribution', ...
                                ['fitting.regularisationType is ''prior'' and fitting.lambda > 0, but ' ...
                                 'extradata.mu/extradata.sigma were not provided to fit().']);
                        end
                        missingMu    = fitting.regmap(~isfield(extradata.mu,    fitting.regmap));
                        missingSigma = fitting.regmap(~isfield(extradata.sigma, fitting.regmap));
                        if ~isempty(missingMu) || ~isempty(missingSigma)
                            error('GACELLE:incompletePriorDistribution', ...
                                ['extradata.mu/extradata.sigma must have a field for every name in ' ...
                                 'fitting.regmap. Missing from mu: {%s}. Missing from sigma: {%s}.'], ...
                                strjoin(missingMu, ', '), strjoin(missingSigma, ', '));
                        end
                        % gpu-convert only the regmap fields actually used by the loss -
                        % NOT this.modelParams, which would silently pull in fields mu/
                        % sigma may not even have (this was a bug in the original
                        % gpuNEXI_priorDistribution.m: it looped over this.modelParams
                        % rather than fitting.regmap)
                        for kreg = 1:numel(fitting.regmap)
                            extradata.mu.(fitting.regmap{kreg})    = gpuArray(single(extradata.mu.(fitting.regmap{kreg})));
                            extradata.sigma.(fitting.regmap{kreg}) = gpuArray(single(extradata.sigma.(fitting.regmap{kreg})));
                        end

                        userFcn      = {@this.FWD};
                        userInput    = {{fitting.lmax, fitting.solver}};
                        userFcn{2}   = @this.prior_distribution_normal;
                        userInput{2} = {mask, fitting.lambda, fitting.regmap, extradata.mu, extradata.sigma};
                        out = askadam().optimisation( dwi, mask, w, pars0, fitting, userFcn, userInput);
                    elseif strcmpi(fitting.dataType,'surface')
                        % custom-regulariser path: userFcn/userInput two-slot form,
                        % see askadam.model_gradient. Volumetric path below is left
                        % on the legacy single-function call form, unchanged.
                        userFcn   = {@this.FWD};
                        userInput = {{fitting.lmax, fitting.solver}};
                        if fitting.lambda{1} > 0
                            if isempty(extradata) || ~isfield(extradata,'neighbours') || ~isfield(extradata,'dr')
                                error('GACELLE:missingSurfaceGeometry', ...
                                    ['fitting.dataType is ''surface'' and fitting.lambda > 0, but ' ...
                                     'extradata.neighbours/extradata.dr were not provided to fit(). ' ...
                                     'This is normally populated by estimate() - if calling fit() ' ...
                                     'directly, supply extradata.neighbours and extradata.dr yourself.']);
                            end
                            userFcn{2}   = @this.surface_total_variation;
                            userInput{2} = {mask, extradata.neighbours, fitting.lambda, fitting.regmap, extradata.dr};
                        end
                        out = askadam().optimisation( dwi, mask, w, pars0, fitting, userFcn, userInput);
                    else
                        out = askadam().optimisation( dwi, mask, w, pars0, fitting, @this.FWD, fitting.lmax, fitting.solver);
                    end
                case 'mcmc'
                    fitting.xStepSize = this.step;
                    
                    out         = mcmc().optimisation(dwi, mask, w, pars0, fitting, @this.FWD, fitting.lmax, fitting.solver);
            end

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            disp('The estimation is completed.');
            
            % clear GPU
            reset(gpuDevice)
            
        end

        %% Data preparation

        % compute weights for optimisation
        function w = compute_optimisation_weights(this,mask,lossFunction,lmax)
        % 
        % Output
        % ------
        % w         : 1D signal masked wegiths
        %

            dims = size(mask,1:3);
            % lmax dependent weights
            l = 0:2:lmax;
            w = zeros([dims numel(this.b)*numel(l)],'single');
            % w = zeros(dims,'single');
            for kl = 1:(lmax/2+1)
                for kb = 1:numel(this.b)
                    w(:,:,:,(kl-1)*numel(this.b)+kb) = this.Nav(kb) / (2*l(kl)+1);
                end
            end
            % if L1 then take square root
            if strcmpi(lossFunction,'l1')
                w = sqrt(w);
            end
            w = w ./ max(w(:));
        end

        % compute rotationally invariant DWI signal if necessary
        function [data, mask, extradata] = prepare_dwi_data(this,data,mask,extradata,lmax,fitting)

            if strcmpi(fitting.dataType,'surface')
                %%%%%%%%%%%% Surface path %%%%%%%%%%%%
                % Expected layout: [1, Nvertex, Nhemi] for both data and mask (singleton
                % 1st dim). No overlapping-slice/halo machinery here: unlike volumetric
                % z-slicing, a hemisphere split has no vertices straddling the cut (each
                % hemisphere's mesh graph is independent by construction), so a split
                % along dim3 is exact rather than an approximation needing halo repair.
                % Arbitrary splits are NOT supported - they would cut real mesh
                % connectivity - so segmentation here is opt-in and exact-only.

                if size(mask,1) ~= 1 || size(data,1) ~= 1
                    error('GACELLE:surfaceShapeMismatch', ...
                        ['Surface mode expects data/mask in [1, Nvertex, Nhemi] convention ' ...
                         '(singleton 1st dimension), got mask size [%s]. See documentation ' ...
                         'for the surface data layout.'], num2str(size(mask,1:3)));
                end
                Nhemi = size(mask,3);
                
                if strcmpi(fitting.regularisationType,'TV') && fitting.lambda{1} > 0
                    % Only surface_total_variation needs mesh connectivity
                    % (neighbours/dr). Prior-distribution regularisation needs
                    % mu/sigma instead (validated later, in the dispatch section)
                    % and has nothing to do with mesh geometry, so this block must
                    % NOT trigger just because lambda{1}>0 - that was true when
                    % 'TV' was the only regularisation option, but is no longer a
                    % safe proxy for "this fit needs surf_dir/hemisphere" now that
                    % 'prior' exists as an alternative gated on the same lambda.
                    if isempty(extradata) || ~isfield(extradata,'surf_dir') || isempty(extradata.surf_dir)
                        error('GACELLE:missingSurfDir', ...
                            'fitting.dataType is ''surface'' but extradata.surf_dir was not provided.');
                    end
                    if ~isfield(extradata,'hemisphere') || isempty(extradata.hemisphere)
                        error('GACELLE:missingHemisphere', ...
                            'fitting.dataType is ''surface'' but extradata.hemisphere was not provided.');
                    end
                    hemisphere = extradata.hemisphere;
                    if ~iscell(hemisphere); hemisphere = cellstr(hemisphere); end
                    if numel(hemisphere) ~= Nhemi
                        error('GACELLE:hemisphereMaskMismatch', ...
                            ['extradata.hemisphere has %d entries but mask dim3 (Nhemi) = %d; ' ...
                             'these must match.'], numel(hemisphere), Nhemi);
                    end
                    depth = 0.5;
                    if isfield(extradata,'depth') && ~isempty(extradata.depth); depth = extradata.depth; end
    
                    [neighbours, dr] = this.get_surface_geometry(extradata.surf_dir, depth, hemisphere);
                    if size(neighbours,3) ~= Nhemi
                        error('GACELLE:surfaceGeometryMismatch', ...
                            ['Computed surface neighbour geometry has %d hemisphere(s) but mask ' ...
                             'has %d; check extradata.hemisphere matches the data.'], ...
                            size(neighbours,3), Nhemi);
                    end
                    extradata.neighbours = neighbours;
                    extradata.dr         = dr;
                end

                % --- segmentation: exact-only, opt-in, {1, Nhemi} legal ---
                NSegmentUser = fitting.NSegmentUser;
                if isempty(NSegmentUser); NSegmentUser = 1; end
                if ~ismember(NSegmentUser, [1, Nhemi])
                    error('GACELLE:invalidSurfaceSegmentation', ...
                        ['Surface segmentation is exact-only and limited to fitting.NSegmentUser ' ...
                         '= 1 (no split) or %d (exact per-hemisphere split); got %d. Arbitrary ' ...
                         'splits would cut mesh connectivity and are not supported.'], Nhemi, NSegmentUser);
                end
            end

            % --- Step 1: compute rotationally invariant signal if needed ---
            if size(data,4)/(lmax/2+1) > numel(this.b) 
                % compute spherical mean signal
                fprintf('Computing rotationally invariant signal...')

                % if the inout little delta is one value then create a vector
                % if isscalar(extradata.ldelta)
                %     extradata.ldelta = ones(size(extradata.bval)) * extradata.ldelta;
                % end
                DWIutilityObj   = DWIutility();
                [data]          = DWIutilityObj.compute_rotationally_invariant_signal(data,extradata.bval,extradata.bvec,[],extradata.BDELTA,[],lmax);
                % [dwi]   = DWIutilityObj.get_Sl_all(dwi,extradata.bval,extradata.bvec,extradata.ldelta,extradata.BDELTA,lmax);

                fprintf('done.\n');

            elseif size(data,4) < numel(this.b) * (lmax/2+1)
                error('GACELLE:inputMismatch', ...
                    'Input has %d volumes but model expects %d. Check lmax or input data.', ...
                    size(data,4), numel(this.b)*(lmax/2+1));
            end

            if ~strcmpi(fitting.dataType,'surface')
                % only for volumetric data

                % --- Step 2: exclude biophysically impossible signal ---
                % |Sl0| > 1 + tolerance is impossible after normalisation by b=0
                % works for both magnitude and real-valued data
                Nshells         = numel(this.b);
                dwi_Sl0         = data(:,:,:,1:Nshells);          % Sl0 block
                mask_impossible = any(abs(dwi_Sl0) > 1 + this.thres_impossible, 4);
                mask_valid      = ~mask_impossible;
    
                % --- Step 3: exclude near-zero signal (background voxels) ---
                % Sl0 of lowest b-value shell should be well above zero for tissue
                % very small value indicates background noise with no diffusion signal
                mask_background = dwi_Sl0(:,:,:,1) < this.thres_bkg;  % first shell = lowest b
                mask_valid      = mask_valid & ~mask_background;
    
                % --- Step 4: exclude incoherent signal (random noise pattern) ---
                % correlate each voxel's Sl0 signal with the median tissue template
                % median is more robust to outliers than mean
                % low correlation indicates random noise rather than coherent diffusion decay
                dwi_2D         = utils.reshape_ND2GD(dwi_Sl0, mask_valid);
                if size(dwi_2D, 2) > 0
                    signalTemplate = median(dwi_2D, 2,'omitmissing');           % median across voxels
                    signalTemplate = (signalTemplate - mean(signalTemplate,'omitmissing')) ./ ...
                                      std(signalTemplate,'omitmissing');
            
                    Rcorr = zeros(1, size(dwi_2D,2));
                    for k = 1:size(dwi_2D,2)
                        signalVoxel = dwi_2D(:,k);
                        denom       = std(signalVoxel);
                        if denom < eps
                            Rcorr(k) = 0;   % flat signal -> zero correlation
                        else
                            signalVoxel = (signalVoxel - mean(signalVoxel)) ./ denom;
                            Rcorr(k)    = corr(signalTemplate, signalVoxel);
                        end
                    end
            
                    Rcorr           = utils.reshape_GD2ND(Rcorr, mask_valid);
                    mask_incoherent = Rcorr < this.thres_similarity;
                    mask_valid      = mask_valid & ~mask_incoherent;
                end
    
                % --- Step 5: remove NaN/Inf ---
                [data,mask_naninf] = utils.remove_img_naninf(data,mask);
                mask_naninf        = max(mask_naninf, [], 4);
                mask_valid         = mask_valid & mask_naninf;
    
                % --- Report and update mask ---
                Nexcluded = sum(mask(:)) - sum(mask_valid(:));
                if Nexcluded > 0
                    fprintf('Signal mask updated: %d voxels excluded (%.1f%% of original mask).\n', ...
                        Nexcluded, 100*Nexcluded/sum(mask(:)));
                    fprintf('  NaN/Inf        : %d\n', sum(mask_naninf(:) & mask(:)));
                    fprintf('  Impossible     : %d\n', sum(mask_impossible(:) & mask(:)));
                    fprintf('  Background     : %d\n', sum(mask_background(:) & mask(:)));
                    fprintf('  Incoherent     : %d\n', sum(mask_incoherent(:) & mask(:)));
                    disp('Please use the updated mask in subsequent analysis.');
                    mask = mask_valid;
                end
            end

        end

        %%%%% Prior estimation related functions %%%%%

        % determine how the starting points will be set up
        function x0 = determine_x0(this,y,mask,fitting) 

            disp('---------------');
            disp('Starting points');
            disp('---------------');

            dims = size(mask,1:3);

            if ischar(fitting.start)
                switch lower(fitting.start)
                    case 'likelihood'
                        % using maximum likelihood method to estimate starting points
                        x0 = this.estimate_prior(y,mask,[],fitting.lmax);
    
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

            fprintf('Estimation lower bound [%s]: [%s]\n',      cell2str(this.modelParams),replace(num2str(fitting.lb(:).',' %.2f'),' ',','));
            fprintf('Estimation upper bound [%s]: [%s]\n',      cell2str(this.modelParams),replace(num2str(fitting.ub(:).',' %.2f'),'  ',','));
            ('---------------');
        end

        % using maximum likelihood method to estimate starting points
        function pars0 = estimate_prior(this,dwi,mask, Nsample,lmax)
        % Estimation starting points for NEXI using likehood method

            rng(this.seed,'twister'); % for reproducible dictionary

            start = tic;
            
            disp('Estimate starting points based on likelihood ...')

            if nargin < 4 || isempty(Nsample)
                Nsample         = 1e4;
            end
            % create training data
            [x_train, S_train] = this.traindata(Nsample,lmax);

            % reshape input data,  put DWI dimension to 1st dim
            dims    = size(dwi);
            dwi     = permute(dwi,[4 1 2 3]);
            dwi     = reshape(dwi,[dims(4), prod(dims(1:3))]);

            % find masked voxels
            ind         = find(mask(:));
            
            Nparam = numel(this.modelParams) - any(strcmpi(this.modelParams,'noise'));

            % manage pool
            pool            = gcp('nocreate');
            isDeletepool    = false;
            if numel(mask(mask>0)) > 1e4    % only start a pool if many voxel
                if isempty(pool)
                    Nworker = min(max(8,floor(maxNumCompThreads/4)),maxNumCompThreads);
                    pool    = parpool('Processes',Nworker);
                    isDeletepool = true;
                end
            end

            pars0_mask  = zeros(Nparam,length(ind));
            if ~isempty(pool)
                parfor kvol = 1:length(ind)
                    pars0_mask(:,kvol) = this.likelihood(dwi(:,ind(kvol)), x_train, S_train,lmax);
                end
            else
                for kvol = 1:length(ind)
                    pars0_mask(:,kvol) = this.likelihood(dwi(:,ind(kvol)), x_train, S_train,lmax);
                end
            end
            pars           = zeros(Nparam,size(dwi,2));
            pars(:,ind)    = pars0_mask;

            % reshape estimation into image
            pars           = permute(reshape(pars,[size(pars,1) dims(1:3)]),[2 3 4 1]);

            % Correction for CSF
            bval_thres      = max(min(gather(this.b)),1.1);
            idx             = gather(this.b) <= bval_thres;
            D0              = real(this.b(idx)\-log(dwi(cat(1,idx,false(size(idx))),:)));
            D0              = permute(reshape(D0,[size(D0,1) dims(1:3)]),[2 3 4 1]);
            D0              = max(utils.set_nan_inf_zero(D0),0);
            mask_CSF        = D0>1.5;
            
            % ratio to modulate pars0 estimattion
            pars0_csf = [0.01,1,1,0.01,0.01];
            for k = 1:size(pars,4)
                tmp                 = pars(:,:,:,k);
                tmp(mask_CSF==1)    = tmp(mask_CSF==1).*pars0_csf(k);
                pars(:,:,:,k)      = tmp;
            end

            ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
            fprintf('Starting points estimated. Elapsed time (hh:mm:ss): %s \n',string(ET));
            if isDeletepool
                delete(pool);
            end

            for km = 1:size(pars,4)
                pars0.(this.modelParams{km}) = pars(:,:,:,km); ...
            end

            % noise
            if strcmpi(this.modelParams{end},'noise')
                pars0.(this.modelParams{end}) = single(ones(size(mask)) * this.startPoint(end));
            end

        end

        % create training data for likelihood
        function [x_train, S_train, intervals] = traindata(this, N_samples, lmax, varargin)
            if nargin < 4
                intervals = [0.01 0.99  ;   % fa
                              1.5 3     ;   % Da
                              0.5 1.5   ;   % De
                                1 100   ;   % exchange time = (1-fa)/r
                             0.01 0.99 ];   % p2
            else
                intervals = varargin{1};
            end
            
            if lmax == 0
                numBSample = numel(this.b);
                numParam   = size(intervals,1) - 1;
            elseif lmax == 2
                numBSample = numel(this.b)*2;
                numParam   = size(intervals,1);
            end
            
            % batch size can be modified according to available hardware
            batch_size  = 1e3;
            reps        = ceil(N_samples/batch_size);
            x_train     = zeros(numParam,batch_size,reps);
            S_train     = zeros(numBSample,batch_size,reps);
            for k = 1:reps
                % generate random parameter guesses and construct batch for NN signal evaluation
                pars = intervals(:,1) + diff(intervals,[],2).*rand(size(intervals,1),batch_size);
                % pars(3,:) = pars(2,:).*pars(3,:);
                pars(4,:) = 1./pars(4,:).*(1-pars(1,:));

                % NEXI Krger signal evaluation
                Sl0 = zeros(numel(this.b),batch_size);
                for j = 1:batch_size
                    Sl0(:,j) = this.Sl0(pars(1,j), pars(2,j), pars(3,j), pars(4,j));
                end

                % in case of Sl2
                if lmax == 2
                    Sl2 = zeros(numel(this.b),batch_size);
                    for j = 1:batch_size
                        Sl2(:,j) = this.Sl2(pars(1,j), pars(2,j), pars(3,j), pars(4,j), pars(5,j)) ;
                    end

                else
                    pars(5,:)   = [];
                    Sl2         = [];
                end

                % remaining signals (dot, soma)
                x_train(:,:,k) = pars;
                S_train(:,:,k) = cat(1,Sl0,Sl2);

            end
            % intervals(3,:) = intervals(2,:).*intervals(3,:);
            intervals(4,:) = (1-intervals(1,end:-1:1))./intervals(4,end:-1:1);
            if lmax == 2
                intervals(5,:) = [];
            end
        end
        
        % likelihood
        function [pars_best, sse_best] = likelihood(this, S0, x_train, S_train,lmax)
            wt = kron(this.Nav(:), 1./(2*(0:2:lmax)+1));
            wt = wt(:);
            nL = floor(lmax/2);
            S0 = S0(1:numel(this.b)*(nL+1),:);
            % batch size can be modified according to available hardware
            [Nx, ~, reps] = size(x_train);
            [~, Nv] = size(S0);
            pars_best = zeros(Nx,Nv);
            sse_best  = inf(1, Nv);
            for k = 1:reps
                pars = x_train(:,:,k);
                S    = S_train(:,:,k);
                for i = 1:Nv
                    S0i = S0(:,i);

                    % scale generated signals (fit S0) to input signal
                    sse = sum(wt.*(S0i - (S0i'*S)./dot(S,S).*S).^2);

                    % store best encountered parameter combination
                    [sse_new,best_index] = min(sse);
                    if sse_new<sse_best(i)
                        sse_best(i)    = sse_new;
                        pars_best(:,i) = pars(:,best_index);
                    end
                end
            end
        end

        % segment data based on slice
        function [dataSeg, maskSeg, pars0Seg, extraDataSeg] = slice_segment(this, data, mask, slice, pars0, extraData)

            dataSeg     = data(:,:,slice,:,:,:,:,:,:);
            maskSeg     = mask(:,:,slice);
            if ~isempty(pars0)
                for km = 1:numel(this.modelParams)
                    pars0Seg.(this.modelParams{km}) = pars0.(this.modelParams{km})(:,:,slice); 
                end
            else      
                pars0Seg = [];                 
            end

            if ~isempty(extraData)
                extraDataSeg = this.slice_extradata_recursive(extraData, slice);
            else                                                    
                extraDataSeg = [];                 
            end

        end

        % Slices every field of an extradata struct along dim3 (the segment's
        % slice/hemisphere range), recursing one level into any struct-valued
        % field. This matters for fields like extradata.mu/extradata.sigma
        % (prior-distribution regularisation), which are themselves structs of
        % per-parameter maps (mu.fa, mu.Da, ...): ismatrix() on a scalar struct
        % is always true regardless of what its fields contain, so treating a
        % struct like any other field would silently take the "pass through
        % unsliced" branch and hand every segment the full-volume prior map
        % instead of the segment's slice - a shape mismatch (or worse, a
        % silent misalignment) inside the loss rather than a clear error.
        % Non-struct fields are sliced exactly as before - fully backward
        % compatible for existing extradata (e.g. surface neighbours/dr).
        function out = slice_extradata_recursive(this, in, slice)
            fields = fieldnames(in);
            out    = struct();
            for kfield = 1:numel(fields)
                val = in.(fields{kfield});
                if isstruct(val)
                    out.(fields{kfield}) = this.slice_extradata_recursive(val, slice);
                elseif ~ismatrix(val)
                    out.(fields{kfield}) = val(:,:,slice,:,:,:,:,:,:,:,:);
                else
                    out.(fields{kfield}) = val;
                end
            end
        end

        %% NEXI signal related functions

        % Forward model to generate NEXI signal
        function [s] = FWD(this, pars, lmax, solver)

            if nargin < 4 || isempty(solver)
                solver = [];
            end
        
            fa   = max(pars.fa, this.epsilon); % avoid division by zeros when computing re
            Da   = pars.Da;
            De   = pars.De;
            ra   = pars.ra;
    
            % % Sl0
            % s = this.Sl0(fa, Da, De, ra,solver);
            
            % Sl2
            if lmax == 2
                p2  = pars.p2;
                % s   = cat(1,s,this.Sl2(fa, Da, De, ra, p2));
            else
                p2 = [];
            end

            % Forward model
            s = this.Slmax2(fa, Da, De, ra, p2, solver);

            % make sure s cannot be greater than 1
            s = min(s,1);   % s = [Nb, Nvoxel]
                
        end

        function S = Slmax2(this, fa, Da, De, ra, p2, solver)

            if nargin < 7; solver = []; end
            if ismatrix(fa)
                GDinput = true;
            else
                GDinput = false;
            end

            bval    = permute(this.b,[2 3 4 1]);
            DELTA   = permute(this.Delta, [2 3 4 1]);

            Da      = bval.*Da;
            De      = bval.*De;
            ra      = DELTA.*ra;
            re      = ra.*fa./(1-fa);

            % Trapezoidal's rule replacement
            Nx  = 14;    % NRMSE<0.05% for Nx=14 for Sl0 and 0.5% for Sl2
            x   = zeros([ones(1,ndims(re)), Nx],'like',De); x(:) = linspace(0,1,Nx);

            % Sl0
            if strcmpi(solver,'mcmc') 
                M   = arrayfun(@NEXI_M,x,fa,Da,De,ra,re);

            else
                % askadam
                M = this.M(x, fa, Da, De, ra, re);

            end

            S = gacelle_trapz(M,x(:),ndims(x));

            % Sl2
            if ~isempty(p2)

                if strcmpi(solver,'mcmc') 
                    % M = M.*(3*x.^2-1)/2; 
                    M = arrayfun(@NEXI_MSl2,M,x);
                    % bypass Matlab's trapz for speed
                    Sl2 = gacelle_trapz(M,x(:),ndims(x));
                else
                    % askadam
                    Sl2 = gacelle_trapz(M.*(3*x.^2-1)/2,x(:),ndims(x));
                end
                Sl2 = p2.*abs(Sl2);

                S = cat(4,S,Sl2);

            end

            % make sure the output is NmeasxNvoxel for GD input
            if GDinput
                if isscalar(fa)
                    S  = squeeze(S);
                else
                    S = squeeze(S).';
                end
            else
                if strcmp(solver,'mcmc')
                    S = permute(S,[4 2 3 1]);
                end
            end

        end
        
        % 0th order rotational invariant
        function S = Sl0(this, fa, Da, De, ra, solver)

            if nargin < 6
                solver = [];
            end

            % if isgpuarray(fa)
            %     bval    = gpuArray(single(this.b));
            %     DELTA   = gpuArray(single(this.Delta));
            % else
            bval    = this.b;
            DELTA   = this.Delta;
            % end
            
            Da = bval.*Da;
            De = bval.*De;
            ra = DELTA.*ra;
            re = ra.*fa./(1-fa);
            
            % Trapezoidal's rule replacement
            Nx  = 14;    % NRMSE<0.05% for Nx=14
            x   = zeros([ones(1,ndims(fa)), Nx],'like',bval); x(:) = linspace(0,1,Nx);
            if strcmpi(solver,'mcmc') 
                dx  = x(2) - x(1);
                M   = arrayfun(@NEXI_M,x,fa,Da,De,ra,re);

                if ndims(M) == 3
                    S = sum((M(:,:,2:end) + M(:,:,1:end-1)) * (dx) / 2, ndims(x));
                elseif ndims(M) == 4
                    S = sum((M(:,:,:,2:end) + M(:,:,:,1:end-1)) * (dx) / 2, ndims(x));
                end

            else
                % askadam
                S   = trapz(x(:),this.M(x, fa, Da, De, ra, re),ndims(x));
            end

            % myfun = @(x) this.M(x, fa, Da, De, ra, re);
            % S = integral(myfun, 0, 1, 'AbsTol', 1e-14, 'ArrayValued', true);

        end
        
        % 2nd order rotational invariant
        function S = Sl2(this, fa, Da, De, ra, p2, solver)

            if nargin < 7
                solver = [];
            end

            % if isgpuarray(fa)
            %     bval    = gpuArray(single(this.b));
            %     DELTA   = gpuArray(single(this.Delta));
            % else
            bval = this.b;
            DELTA   = this.Delta;
            % end

            Da = bval.*Da;
            De = bval.*De;
            ra = DELTA.*ra;
            re = ra.*fa./(1-fa);
            
            % Trapezoidal's rule replacement
            Nx  = 14;    % NRMSE<0.5% for Nx=14
            x   = zeros([ones(1,ndims(fa)), Nx],'like',bval); x(:) = linspace(0,1,Nx);

            if strcmpi(solver,'mcmc') 

                dx  = x(2) - x(1);

                % M = M.*(3*x.^2-1)/2; 
                M = arrayfun(@NEXI_MSl2,arrayfun(@NEXI_M,x,fa,Da,De,ra,re),x);
                % bypass Matlab's trapz for speed
                if ndims(M) == 3
                    S = sum((M(:,:,2:end) + M(:,:,1:end-1)) * (dx) / 2, ndims(x));
                elseif ndims(M) == 4
                    S = sum((M(:,:,:,2:end) + M(:,:,:,1:end-1)) * (dx) / 2, ndims(x));
                end
            else
                % askadam
                S   = trapz(x(:),this.M(x, fa, Da, De, ra, re).*(3*x.^2-1)/2,ndims(x));
            end

            S   = p2.*abs(S);

        end

    end

    methods(Static)
        %% NEXI signal related
        function M = M(x, fa, Da, d2, r1, r2)
            d1 = Da.*x.^2;
            l1 = (r1+r2+d1+d2)/2;
            l2 = sqrt( (r1-r2+d1-d2).^2 + 4*r1.*r2 )/2; l2 = max(l2, askadam.epsilon);  % avoid division by zeros
            lm = l1-l2;
            Pp = (fa.*d1 + (1-fa).*d2 - lm)./(l2*2);
            M  = Pp.*exp(-(l1+l2)) + (1-Pp).*exp(-lm); 
        end

        %% Utilities
        % check and set default fitting algorithm parameters
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

                if ~isfield(fitting,'regmap');              fitting2.regmap             = 'fa';             end

                if ~iscell(fitting2.regmap)
                    fitting2.regmap = cellstr(fitting2.regmap);
                end

                % =====================================================================
                % Regularisation type: 'TV' (default) | 'prior'
                % Default is 'TV' (not 'none') to exactly preserve pre-existing
                % behaviour, where any regularisation was implicitly TV/surfaceTV,
                % gated only on fitting.lambda{1}>0. 'none' is still expressed by
                % lambda{1}==0 elsewhere and does not need a separate flag value here.
                % =====================================================================
                if ~isfield(fitting,'regularisationType');  fitting2.regularisationType = 'TV';              end
                if ~ismember(lower(fitting2.regularisationType), {'tv','prior'})
                    error('GACELLE:invalidRegularisationType', ...
                        "fitting.regularisationType must be 'TV' or 'prior', got '%s'.", fitting2.regularisationType);
                end

            end


            % get customised fitting setting check
            if ~isfield(fitting,'lmax');                fitting2.lmax               = 0;                end
            if ~isfield(fitting,'start');               fitting2.start              = 'likelihood';     end

            % =====================================================================
            % Data geometry: 'volumetric' (default) | 'surface'
            % 'surface' bypasses fitting.TVmode entirely - TVmode only selects the
            % axis convention for the built-in volumetric TV operator, and surface
            % regularisation is dispatched through a separate custom-function path
            % (surface_total_variation) regardless of what TVmode is set to.
            % =====================================================================
            if ~isfield(fitting,'dataType');            fitting2.dataType           = 'volumetric';     end
            if ~ismember(lower(fitting2.dataType), {'volumetric','surface'})
                error('GACELLE:invalidDataType', ...
                    "fitting.dataType must be 'volumetric' or 'surface', got '%s'.", fitting2.dataType);
            end
            if strcmpi(fitting2.dataType, 'surface') && ~isempty(fitting2.NSegmentUser)
                if fitting2.NSegmentUser > 2
                    fitting2.NSegmentUser = 2;
                    warning('GACELLE:invalidNSegmentUser', ...
                    "fitting.NSegmentUser must be <= 2 for surface-based estimation, got '%s'. Set to 2 instead.", fitting2.NSegmentUser);
                end
            end
        end

        
        function loss_reg = surface_total_variation(parameters, mask, neighbours, lambda, regmap, edgeLength)
        % loss_reg = surface_total_variation(parameters, mask, neighbours, lambda, regmap, edgeLength)
        %
        % Input
        % --------------
        % parameters    : structure variable containing the model parameters (same as forward model function)
        % mask          : 3D mask, [1,Nvertex,Nhemi]
        % neighbours    : [1,Nvertex,Nhemi,maxNeighbours] per-hemisphere-LOCAL vertex indices
        %                  (see utils.get_surface_neighbours; NOT global linear indices)
        % lambda        : 1D cell array of regularisation parameter
        % regmap        : 1D cell array of the names of the parameter maps where TV applies to
        % edgeLength    : [1,Nvertex,Nhemi,maxNeighbours] scalar Euclidean edge length to each
        %                  neighbour, precomputed once by utils.get_surface_neighbours (fixed
        %                  mesh geometry - not recomputed here on every optimiser iteration)
        %
        % Output
        % --------------
        % loss_reg      : regularisation loss
        %
        % Description: distance-normalised surface total variation regularisation loss,
        % i.e. sum over mesh edges of |theta(vi)-theta(vj)| / ||vi-vj||. Padding entries
        % (self-indexed neighbours, from ragged per-vertex neighbour counts) have
        % ||vi-vj|| = 0 by construction and are excluded from the sum.
        %
        % Note: this is a distance-weighted regulariser, not the unweighted absolute
        % difference stated in Eq. 5 of the manuscript - Eq. 5 needs updating to match
        % if this is the version being reported, and lambda should be re-tuned after
        % this change since the loss scale differs from the unweighted/per-axis forms
        % previously used.
        
        % Kwok-Shing Chan @ MGH
        % kchan2@mgh.harvard.edu
        %
        % Date created: 11 April 2025
        % Date modified: 3 July 2026 (takes precomputed scalar edgeLength directly
        %                 instead of the per-axis dr vector, so the sqrt(sum(dr.^2,5))
        %                 reduction happens once in utils.get_surface_neighbours rather
        %                 than once per optimiser iteration inside this loss function;
        %                 also replaces the earlier per-axis dx/dy/dz division, which
        %                 spuriously produced Inf for any edge that happened to be
        %                 axis-aligned on one coordinate - a real, not-rare occurrence
        %                 on a curved mesh, silently discarding those edges from every
        %                 loss evaluation; fixed neighbour indexing to use global
        %                 linear indices across hemispheres, since neighbours holds
        %                 per-hemisphere-local indices but was being used directly as a
        %                 linear index into the full [1,Nvertex,Nhemi] theta array,
        %                 silently reading hemisphere-1 data for every hemisphere;
        %                 replaced isnan/isinf boolean compaction - whose OUTPUT LENGTH
        %                 varies with how many entries are invalid - with in-place
        %                 zeroing, since dlaccelerate requires every intermediate
        %                 tensor's shape to stay fixed across calls to a cached trace,
        %                 and a shape-varying selection breaks that the moment the
        %                 invalid-entry count differs from the traced call
        %

        % regularisation term
        loss_reg = 0;
        if lambda{1} > 0
            Nsample  = numel(mask(mask ~= 0));
            Nvertex  = size(mask, 2);
            Nhemi    = size(mask, 3);
        
            % neighbours holds per-hemisphere-LOCAL vertex indices (1:Nvertex, reset
            % for every hemisphere) - convert to global linear indices into the full
            % [1,Nvertex,Nhemi] theta array before using them to index theta directly.
            % No-op when Nhemi==1 (e.g. inside a per-hemisphere-segmented fit()).
            hemiOffset       = reshape((0:Nhemi-1) * Nvertex, 1, 1, []);
            neighboursGlobal = neighbours + hemiOffset;
        
            for kreg = 1:numel(lambda)
        
                theta               = utils.reshape_GD2ND(parameters.(regmap{kreg}),mask);
                thetaNeighbours     = zeros(size(neighbours), "like",theta);
                thetaNeighbours(:)  = theta(neighboursGlobal(:));
        
                dthetadr = (theta - thetaNeighbours) ./ edgeLength;
        
                % zero invalid entries in place rather than compacting them out, so
                % size(dthetadr) stays fixed across calls - required for dlaccelerate
                % to safely reuse a cached trace (see header note above)
                invalid              = ~isfinite(dthetadr);
                dthetadr(invalid)    = 0;
                dthetadr             = utils.reshape_ND2GD(dthetadr,mask);
        
                loss_reg = sum(abs(dthetadr(:)))/Nsample*lambda{kreg} + loss_reg;
        
            end
        end
        
        end

        function loss_reg = prior_distribution_normal(parameters, mask, lambda, regmap, mu, sigma)
        % loss_reg = prior_distribution_normal(parameters, mask, lambda, regmap, mu, sigma)
        %
        % Input
        % --------------
        % parameters    : structure variable containing the model parameters (same as forward model function)
        % mask          : signal mask, 3D [x,y,z] (volumetric) or [1,Nvertex,Nhemi] (surface)
        % lambda        : 1D cell array of regularisation parameter
        % regmap        : 1D cell array of the names of the parameter maps where the prior applies to
        % mu            : structure variable containing the mean of model parameters distribution,
        %                  one field per name in regmap, same spatial shape as mask
        % sigma         : structure variable containing the SD of model parameters distribution,
        %                  one field per name in regmap, same spatial shape as mask
        %
        % Output
        % --------------
        % loss_reg      : regularisation loss
        %
        % Description: penalises deviation of each regmap parameter map from an externally
        % supplied (e.g. group-level, registered) mean/SD prior, in z-score units. Geometry
        % agnostic - works for both 'volumetric' and 'surface' fitting.dataType, since
        % utils.reshape_GD2ND/mask handles both mask conventions identically to the forward
        % model. Ported from the standalone prior_distribution_normal.m used previously only
        % via the surface-only gpuNEXI_priorDistribution.m; folded in here so both dataTypes
        % share the same fit()/slice_segment()/segmentation machinery instead of a separate
        % model class. mu/sigma are only ever populated (gpu-converted, sliced per segment)
        % for the fields listed in regmap - see fit() and slice_extradata_recursive().
        %
        % Kwok-Shing Chan @ MGH
        % kchan2@mgh.harvard.edu
        % Date created: 11 April 2025
        % Date modified: 6 July 2026 (folded into gpuNEXI.m as a static method so it shares
        %                 segmentation/slicing with the volumetric and surface paths; fixed
        %                 a caller-side bug where mu/sigma were gpu-converted by looping
        %                 over this.modelParams instead of fitting.regmap - see fit())

        % regularisation term
        loss_reg = 0;
        if lambda{1} > 0
            Nsample = numel(mask(mask ~= 0));

            for kreg = 1:numel(lambda)

                theta  = utils.reshape_GD2ND(parameters.(regmap{kreg}), mask);
                zscore = (theta - mu.(regmap{kreg})) ./ sigma.(regmap{kreg});

                loss_reg = sum(abs(zscore(and(~isnan(zscore),~isinf(zscore)))))/Nsample*lambda{kreg} + loss_reg;

            end
        end

        end

    end

end