classdef gpuR2starMapping < handle
% This is the method to perform R2* mapping using multiecho GRE (mGRE) data
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 5 August 2026
% Date modified: 

    properties (GetAccess = public, SetAccess = protected)
    % ===== MODEL PARAMETER CONTRACT =====
    % M0        : Proton density weighted signal
    % R2star    : R2* in s^-1   
    % noise     : noise
    %
    % modelParams{k} <-> ub(k) <-> lb(k) <-> startPoint(k) <-> step(k)
    % These five arrays MUST stay the same length and index-aligned.
    % Mutate only as a set, via updateProperty() - never assign into a
    % single element from outside the class, or these will desync.
    %
    % 'noise' is solver-conditional (mcmc only) and is kept LAST for
    % index-alignment with ub/lb/startPoint/step. updateProperty() strips
    % it by name via ismember(), not by index, when the solver is not
    % mcmc. Any future solver-conditional parameter should likewise go
    % last.
        modelParams    = {'M0';'R2star';'noise'};
        ub              = [   2;    200;    0.1];
        lb              = [   0;    0.1;  0.001];
        startPoint      = [   1;     30;   0.05];
        step            = [0.01;      1;  0.005];
    end

    properties
    % ===== USER-TUNABLE OPTIONS =====
    % Freely settable by users before fitting; no coupling between these.

        seed = 48463;   % for reproducible random number generation

        epsilon = utils.epsilon;
    end

    properties (GetAccess = public, SetAccess = protected)
    % ===== ACQUISITION PARAMETERS =====
    % Set once in the constructor from user-provided acquisition info.
    % Read-only after construction.
        te;
        B0 = 3;
    end

    properties
        % default model parameters and estimation boundary
    end

    methods

        % constructor
        function this = gpuR2starMapping(te)

            this.te = single(te(:));

        end
        
        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('==========================');
            disp('R2* mapping with mGRE data');
            disp('==========================');

            
            disp('----------------')
            disp('Data Information');
            disp('----------------')
            fprintf('Echo time (TE) (ms)             : [%s] \n',num2str(this.te.' * 1e3,' %.2f'));
            disp('----------------')

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

        end

        %% higher-level data fitting functions
        % Wrapper function of fit to handle image data; automatically segment data and fitting in case the data cannot fit in the GPU in one go
        function  [out] = estimate(this, data, mask, fitting)
        % Perform R2* model parameter estimation
        % Input data are expected in multi-dimensional image format
        % 
        % Input
        % -----------
        % data      : 4D image data, [x,y,z,echoes]
        % mask      : 3D signal mask, [x,y,z]
        % fitting   : fitting algorithm parameters (see fit function)
        % 
        % Output
        % -----------
        % out       : output structure contains all estimation results
        % M0        : T1w weighted (and coil sensitivity) signal
        % R2star    : R2* map
        % 

            % display basic info
            this.display_data_model_info;
            
            % get all fitting algorithm parameters 
            fitting                     = this.check_set_default(fitting);

            % normalised data if needed
            [data, mask, scaleFactor]   = this.prepare_data( data, mask);

            % convert datatype to single
            data    = single(data);
            mask    = mask > 0;

            %%%%%%%%%%%%%%%% Step 2: Memory management %%%%%%%%%%%%%%%%
            
            % --- [Experimental] estimate memory usage using a small batch of data size ---
            % this method tends to be more conservative than the actual memory ussage
            [seg,NSegment] = utils.find_optimal_segment_3D(this, data, mask, fitting);

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
                [dataSeg, maskSeg]              = this.slice_segment(data, mask, fitRange);

                % run fitting
                [outSeg] = this.fit(dataSeg,maskSeg,fitting);

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

                    out.min.M0      = out.min.M0 * scaleFactor; % undo scaling
                    out.final.M0    = out.final.M0 * scaleFactor; % undo scaling

                    askadam.save_askadam_output(fitting.outputFilename,out)
                case 'mcmc'

                    % rescale M0
                    for k = 1:numel(fitting.metric)
                        out.(fitting.metric{k}).M0 = out.(fitting.metric{k}).M0 *scaleFactor;
                    end
                    out.posterior.M0 = out.posterior.M0 *scaleFactor;
                    mcmc.save_mcmc_output(fitting.outputFilename,out)
            end

        end

        % Data fitting function, can be 2D (voxel-based) or 4D (image-based)
        function [out] = fit(this, data, mask, fitting)
        %
        % Input
        % -----------
        % data      : multi-echo data images, [x,y,z,TE]
        % mask      : 3D signal mask, [x,y,z]
        % fitting   : fitting algorithm parameters
        % 
        % Output
        % -----------
        % out       : output structure
        %   .final      : final results
        %       .loss       : final loss metric
        %   .min        : results with the minimum loss metric across all iterations
        %       .loss       : loss metric      
        %
        % Kwok-Shing Chan @ MGH
        % kchan2@mgh.harvard.edu
        %
            
            % get image size
            dims = size(data,1:3);

            %%%%%%%%%%%%%%%%%%%% 1. Validate and parse input %%%%%%%%%%%%%%%%%%%%
            if nargin < 3 || isempty(mask); mask = ones(dims,'logical'); end % if no mask input then fit everthing
            if nargin < 4; fitting = struct(); end

            % get all fitting algorithm parameters 
            fitting                 = this.check_set_default(fitting);
            % determine fitting parameters
            this                    = this.updateProperty(fitting);
            fitting.modelParams     = this.modelParams;
            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(this.modelParams)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(this.modelParams)); end

            % set initial tarting points
            pars0 = this.determine_x0(data,mask,fitting) ;

            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w = this.compute_optimisation_weights(data,fitting); % This is a tailored funtion

            % 2.2 display optimisation algorithm parameters
            this.display_algorithm_info(fitting)


            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            % 2.3 askAdam optimisation main
            switch fitting.solver
                case 'askadam'

                    out         = askadam().optimisation(data, mask, w, pars0, fitting, @this.FWD, fitting.solver);

                case 'mcmc'
                    fitting.xStepSize = this.step;

                    % initial global optimisation
                    out         = mcmc().optimisation(data, mask, w, pars0, fitting, @this.FWD, fitting.solver, fitting);
                    
            end

            disp('The process is completed.')
            
            % clear GPU
            reset(gpuDevice)
            
        end
        
        %% Prior estimation related functions

        % determine how the starting points will be set up
        function x0 = determine_x0(this,y,mask,fitting) 

            disp('---------------');
            disp('Starting points');
            disp('---------------');

            dims = size(mask,1:3);

            if ischar(fitting.start)
                switch lower(fitting.start)
                    case 'prior'
                        % using maximum likelihood method to estimate starting points
                        x0 = this.estimate_prior(y);
    
                    case 'default'
                        % use fixed points
                        fprintf('Using default starting points for all voxels at [%s]: [%s]\n', cell2str(this.modelParams),replace(num2str(this.startPoint(:).',' %.2f'),' ',','));
                        x0 = utils.initialise_x0(dims,this.modelParams,this.startPoint);

                end
            else
                % user defined starting point
                fprintf('Using user-defined starting points for all voxels at [%s]: [%s]\n',cell2str(this.modelParams),replace(num2str(fitting.start(:).',' %.2f'),' ',','));
                x0 = utils.initialise_x0(dims,this.modelParams,fitting.start);

            end
            
            % make sure the input is bounded
            x0 = askadam.set_boundary(x0,fitting.ub,fitting.lb);

            fprintf('Estimation lower bound [%s]: [%s]\n',      cell2str(this.modelParams),replace(num2str(fitting.lb(:).',' %.2f'),' ',','));
            fprintf('Estimation upper bound [%s]: [%s]\n',      cell2str(this.modelParams),replace(num2str(fitting.ub(:).',' %.2f'),'  ',','));
            ('---------------');
        end

        % closed-form solution to estimate better starting points
        function pars0 = estimate_prior(this,data)

            dims = size(data,1:3);

            for k = 1:numel(this.modelParams)
                pars0.(this.modelParams{k}) = single(this.startPoint(k)*ones(dims));
            end

            disp('Estimate starting points using closed-form solutions...')
            
            start = tic;
            % R2* closed-form solution
            R2s0            = this.R2star_trapezoidal(mean(abs(data),5),this.te);
            mask_valid = and(~isnan(R2s0),~isinf(R2s0));
            R2s0(mask_valid == 0) = this.lb(3); R2s0(R2s0<this.lb(2)) = this.lb(3); R2s0(R2s0>this.ub(2)) = this.ub(2);

            % always follow the order specified in the beginning of the file
            pars0.(this.modelParams{1}) = single(abs(data(:,:,:,1))); 
            pars0.(this.modelParams{2}) = single(R2s0);

            ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
            fprintf('Starting points estimated. Elapsed time (hh:mm:ss): %s \n',string(ET));
  
        end

        %% Signal related functions

        % compute the forward model
        function [s] = FWD(this, pars, solver, fitting)

            if nargin < 3
                solver = [];
            end
            if nargin < 4
                fitting = [];
            end
            
            TE  =  permute(this.te,[2 3 4 1]); % TE in 4th dimension
            
            M0      = pars.M0;
            R2star  = pars.R2star;
            
            if strcmpi(solver, 'mcmc')

                % MCMC

                s = arrayfun(@model_R2s_singlecompartment,M0, R2star, TE);

                % vectorise to match masked measurement data
                s = utils.reshape_ND2GD(s,[]);
                % reshape s for ensemble solver
                if ~isempty(fitting)
                    if strcmpi(fitting.algorithm,'ensemble')
                        s = reshape(s, [size(s,1) size(s,2)/fitting.Nwalker fitting.Nwalker]);
                    end
                end

            else

                % askadam

                s = this.model_R2s(M0, R2star, TE);

                if ismatrix(M0)
                    % vectorise to match maksed measurement data
                    s = utils.reshape_ND2GD(s,[]);
                end

            end
                
        end
        
        %% utility
        
        % make sure input data are valid
        function [mask] = validate_input(this,data,mask)
           
            %%%%%%%%%% 2. check data integrity %%%%%%%%%%
            disp('-----------------------');
            disp('Checking data integrity');
            disp('-----------------------');

            % check if the number of echo times matches with the data
            if numel(this.te) ~= size(data,4)
                error('The size of TE does not match with the 4th dimension of the image.');
            end

            % check signal mask
            if ~isempty(mask)
                disp('Mask input                : True');
                if ~isequal(size(data,1:3), size(mask,1:3))
                    error('The dimension of the mask does not match the inpt image.');
                end
            else
                disp('Mask input                : False');
                disp('Default masking method is used.');

                % if no mask input then use default method to generate a signal mask
                mask = max(abs(data),[],4)./max(abs(data(:))) > 0.05;
            end

            disp('Input data is valid.')
        end

        % normalise input data based on masked signal intensity at 98%
        function [img, mask, scaleFactor] = prepare_data(this, img, mask)
            
            mask = mask>0;

            % make sure input data are valid
            [mask] = this.validate_input(img,mask);

            tmp = abs(img(:,:,:,1));

            scaleFactor = prctile( tmp(mask>0), 98);

            img = img ./ scaleFactor;

            % mask sure no nan or inf
            [img,mask] = utils.remove_img_naninf(img,mask);

        end

        % segment data based on slice
        function [dataSeg, maskSeg] = slice_segment(this, data, mask, slice)

            dataSeg     = data(:,:,slice,:,:,:,:,:,:);
            maskSeg     = mask(:,:,slice);
        end

    end

    methods(Static)
        %% Signal related
        function signal = model_R2s(m0,r2s,te)
        % m0    : proton density weighted signal
        % r2s   : R2*, in s^-1 or ms^-1
        % te    : echo time, in s or ms

            signal = m0 .* exp(-te .* r2s);
        
        end

        function [R2star,S0] = R2star_trapezoidal(img,te)
            % disgard phase information
            img = double(abs(img));
            te  = double(te);
            
            dims = size(img);
            
            % main
            % Trapezoidal approximation of integration
            temp = 0;
            for k=1:dims(4)-1
                temp = temp+0.5*(img(:,:,:,k)+img(:,:,:,k+1))*(te(k+1)-te(k));
            end
            
            % very fast estimation
            t2s = temp./(img(:,:,:,1)-img(:,:,:,end));
                
            R2star = 1./t2s;

            S0 = img(1:(numel(img)/dims(end)))'.*exp(R2star(:)*te(1));
            if numel(S0) ~=1
                S0 = reshape(S0,dims(1:end-1));
            end
        end

        %% Utilities
        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default(fitting)

            if ~isfield(fitting,'solver');      fitting.solver = 'askadam';        end

            % get basic fitting setting check
            if strcmpi(fitting.solver,'mcmc')

                % mcmc
                fitting2 = mcmc.check_set_default_basic(fitting);

            else

                % askadam
                fitting2 = askadam.check_set_default_basic(fitting);

            end

            % get customised fitting setting check
            if ~isfield(fitting,'weightMethod');        fitting2.weightMethod   = '1stecho';        end
            if ~isfield(fitting,'isWeighted');          fitting2.isWeighted     = false;            end
            if ~isfield(fitting,'weightPower');         fitting2.weightPower    = 2;                end
            if ~isfield(fitting,'start');               fitting2.start          = 'prior';          end

        end

        % display fitting algorithm information
        function display_algorithm_info(fitting)
        
            % You may add more dispay messages here
            disp('Cost function options:');
            if fitting.isWeighted
                disp('Cost function weighted by echo intensity: True');
                disp(['Weighting method: ' fitting.weightMethod]);
                if strcmpi(fitting.weightMethod,'1stEcho')
                    disp(['Weighting power: ' num2str(fitting.weightPower)]);
                end
            else
                disp('Cost function weighted by echo intensity: False');
            end
            
            
        end

        % compute weights for optimisation
        function w = compute_optimisation_weights(data,fitting)
        % 
        % Output
        % ------
        % w         : 1D signal masked wegiths
        %
            % weights
            if fitting.isWeighted
                switch lower(fitting.weightMethod)
                    case 'norm'
                        % weights using echo intensity, as suggested in Nam's paper
                        w = sqrt(abs(data));
                    case '1stecho'
                        p = fitting.weightPower;
                        % weights using the 1st echo intensity 
                        w = bsxfun(@rdivide,abs(data).^p,abs(data(:,:,:,1)).^p);
                end
            else
                % compute the cost without weights
                w = ones(size(data),'single');
            end
            w(w>1) = 1; w(w<0) = 0;
        end

    end

end