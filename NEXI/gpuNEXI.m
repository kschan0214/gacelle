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
        % fitting   : fitting algorithm parameters (see fit function)
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
            [data,mask] = this.prepare_dwi_data(data,mask,extradata,fitting.lmax);

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
                fitRange                        = seg(kseg).fit;
                ownedRange                      = seg(kseg).owned;
                [dataSeg, maskSeg, pars0Seg]    = this.slice_segment(data, mask, fitRange, pars0);

                % run fitting
                [outSeg] = this.fit(dataSeg,maskSeg,fitting,pars0Seg);

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

        % Data fitting function, can be 2D (voxel-based) or 4D (image-based)
        function [out] = fit(this,dwi,mask,fitting,pars0)
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
            
            % check GPU
            gpool = gpuDevice;
            
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

            % get all fitting algorithm parameters 
            fitting                 = this.check_set_default(fitting);
            % determine fitting parameters
            this                    = this.updateProperty(fitting);
            fitting.modelParams     = this.modelParams;
            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(fitting.modelParams)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(fitting.modelParams)); end
            
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
                    askadamObj  = askadam();
                    out         = askadamObj.optimisation( dwi, mask, w, pars0, fitting, @this.FWD, fitting.lmax, fitting.solver);
                case 'mcmc'
                    fitting.xStepSize = this.step;
                    
                    mcmcObj     = mcmc(); 
                    out         = mcmcObj.optimisation(dwi, mask, w, pars0, fitting, @this.FWD, fitting.lmax, fitting.solver);
            end

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            disp('The estimation is completed.');
            
            % clear GPU
            reset(gpool)
            
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
        function [dwi, mask] = prepare_dwi_data(this,dwi,mask,extradata,lmax)

            % --- Step 1: compute rotationally invariant signal if needed ---
            if size(dwi,4)/(lmax/2+1) > numel(this.b) 
                % compute spherical mean signal
                fprintf('Computing rotationally invariant signal...')

                % if the inout little delta is one value then create a vector
                % if isscalar(extradata.ldelta)
                %     extradata.ldelta = ones(size(extradata.bval)) * extradata.ldelta;
                % end
                DWIutilityObj   = DWIutility();
                [dwi]           = DWIutilityObj.compute_rotationally_invariant_signal(dwi,extradata.bval,extradata.bvec,[],extradata.BDELTA,[],lmax);
                % [dwi]   = DWIutilityObj.get_Sl_all(dwi,extradata.bval,extradata.bvec,extradata.ldelta,extradata.BDELTA,lmax);

                fprintf('done.\n');

            elseif size(dwi,4) < numel(this.b) * (lmax/2+1)
                error('GACELLE:inputMismatch', ...
                    'Input has %d volumes but model expects %d. Check lmax or input data.', ...
                    size(dwi,4), numel(this.b)*(lmax/2+1));
            end

            % --- Step 2: exclude biophysically impossible signal ---
            % |Sl0| > 1 + tolerance is impossible after normalisation by b=0
            % works for both magnitude and real-valued data
            Nshells         = numel(this.b);
            dwi_Sl0         = dwi(:,:,:,1:Nshells);          % Sl0 block
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
            [dwi,mask_naninf] = utils.remove_img_naninf(dwi,mask);
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

            rng(this.seed); % for reproducible dictionary

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
        function [dataSeg, maskSeg, pars0Seg] = slice_segment(this, data, mask, slice, pars0)

            dataSeg     = data(:,:,slice,:,:,:,:,:,:);
            maskSeg     = mask(:,:,slice);
            if ~isempty(pars0)
                for km = 1:numel(this.modelParams)
                    pars0Seg.(this.modelParams{km}) = pars0.(this.modelParams{km})(:,:,slice); 
                end
            else      
                pars0Seg = [];                 
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

            bval    = this.b;
            DELTA   = this.Delta;

            Da      = bval.*Da;
            De      = bval.*De;
            ra      = DELTA.*ra;
            re      = ra.*fa./(1-fa);

            % Trapezoidal's rule replacement
            Nx  = 14;    % NRMSE<0.05% for Nx=14 for Sl0 and 0.5% for Sl2
            x   = zeros([ones(1,ndims(fa)), Nx],'like',De); x(:) = linspace(0,1,Nx);
            % x   = zeros([ones(1,ndims(fa)), Nx], 'single', 'gpuArray'); x(:) = linspace(0,1,Nx);

            % Sl0
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
                M = this.M(x, fa, Da, De, ra, re);

                % S = trapz(x(:),M,ndims(x));
                S = gacelle_trapz(M,x(:),ndims(x));
                
            end

            % Sl2
            if ~isempty(p2)

                if strcmpi(solver,'mcmc') 
                    % M = M.*(3*x.^2-1)/2; 
                    M = arrayfun(@NEXI_MSl2,M,x);
                    % bypass Matlab's trapz for speed
                    if ndims(M) == 3
                        Sl2 = sum((M(:,:,2:end) + M(:,:,1:end-1)) * (dx) / 2, ndims(x));
                    elseif ndims(M) == 4
                        Sl2 = sum((M(:,:,:,2:end) + M(:,:,:,1:end-1)) * (dx) / 2, ndims(x));
                    end
                else
                    % askadam
                    % Sl2 = trapz(x(:),M.*(3*x.^2-1)/2,ndims(x));
                    Sl2 = gacelle_trapz(M.*(3*x.^2-1)/2,x(:),ndims(x));
                end
                Sl2 = p2.*abs(Sl2);

                S = cat(1,S,Sl2);
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

            end


            % get customised fitting setting check
            if ~isfield(fitting,'lmax');                fitting2.lmax               = 0;                end
            if ~isfield(fitting,'start');               fitting2.start              = 'likelihood';     end

            

        end

    end

end