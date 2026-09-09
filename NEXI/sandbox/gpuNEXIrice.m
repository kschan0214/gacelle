classdef gpuNEXIrice < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 8 Dec 2023 (v0.1.0)
% Date modified: 29 March 2024 (v0.2.0)
% Date modified: 4 April 2024 (v0.3.0)
% Date modified: 20 August 2024 (v0.4.0)

    properties
        % default model parameters and estimation boundary
        % fa        : Neurite volume fraction
        % Da        : longitudinal diffusivity of neurite [ms/us^2]
        % De        : diffusivity of extracellular water [ms/us^2]
        % ra        : exchange rate from neurite to extracellular space [1/s]
        % p2        : non-linear neurite dispersion index
        modelParams     = {'fa';'Da';'De';'ra';'p2'};
        ub              = [1-eps;   3;   3;   1;  1];
        lb              = [ eps; eps; eps;1/250; eps];
        startPoint      = [ 0.4;   2;   1; 0.05; 0.2];
    end

    properties (GetAccess = public, SetAccess = protected)
        b;
        Delta;  
        Nav;
    end
    
    methods

        % constructuor
        function this = gpuNEXIrice(b, Delta, varargin)
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
        % Usage
        % ----------
        % obj                   = NEXI(b, Delta, Nav);
        % [out, fa, Da, De, r]  = obj.fit(S, mask, fitting,);
        % Sfit                  = smt.FWD([fa, Da, De, r]);
        % [x_train, S_train]    = obj.traindata(1e4);
        % pars0                 = smt.likelihood(S, x_train, S_train);
        % [out, fa, Da, De, r]  = smt.fit(S, mask, fitting, pars0);
        %
        % Author:
        %  Kwok-Shing Chan (kchan2@mgh.harvard.edu) 
        %  Hong-Hsi Lee (hlee84@mgh.harvard.edu)
        %  Copyright (c) 2023 Massachusetts General Hospital
        %
        %  Adapted from the code of
        %  Dmitry Novikov (dmitry.novikov@nyulangone.org)
        %  Copyright (c) 2023 New York University
            
            this.b      = b(:) ;
            this.Delta  = Delta(:) ;
            if nargin > 2
                this.Nav = varargin{1} ;
            else
                this.Nav =  ones(size(b)) ;
            end
            this.Nav = this.Nav(:) ;
        end
        
        % update properties according to lmax
        function this = updateProperty(this, fitting)

            % DIMWI
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

            disp('=============================');
            disp('NEXI Rice with askAdam solver');
            disp('=============================');

            disp('----------------')
            disp('Data Information');
            disp('----------------')
            fprintf('b-shells (ms/um2)              : [%s] \n',num2str(this.b.',' %.2f'));
            fprintf('Diffusion time (ms)            : [%s] \n',num2str(this.Delta.',' %i'));
            disp('----------------');
        end

        %% higher-level data fitting functions
        % Wrapper function of fit to handle image data; automatically segment data and fitting in case the data cannot fit in the GPU in one go
        function  [out] = estimate(this, dwi, mask, extraData, fitting, pars0)
        % Perform NEXI model parameter estimation based on askAdam
        % Input data are expected in multi-dimensional image
        % 
        % Input
        % -----------
        % dwi       : 4D DWI, [x,y,z,dwi]
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
            fitting = this.check_set_default(fitting);

            % get matrix size
            dims = size(dwi,1:3);

            %%%%%%%%%%%%%%%% Step 1: Validate all input data %%%%%%%%%%%%%%%%
            % compute rotationally invariant signal if needed
            [dwi,mask,extraData] = this.prepare_dwi_data(dwi,mask,extraData,fitting.lmax);

            % mask sure no nan or inf
            [dwi,mask] = utils.remove_img_naninf(dwi,mask);

            % if no pars input at all (not even empty) then use prior
            if nargin < 6; pars0 = []; end

            % convert datatype to single
            dwi     = single(dwi);
            mask    = mask >0;
            if ~isempty(pars0); for km = 1:numel(this.modelParams); pars0.(this.modelParams{km}) = single(pars0.(this.modelParams{km})); end; end

            %%%%%%%%%%%%%%%% End Step 1 %%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%% Step 2: Validate if GPU has enough memory  %%%%%%%%%%%%%%%%
            % determine if we need to divide the data to fit in GPU
            g = gpuDevice; reset(g);
            memoryFixPerVoxel       = 0.0013;   % get this number based on mdl fit
            memoryDynamicPerVoxel   = 0.05;     % get this number based on mdl fit
            [NSegment,maxSlice]     = utils.find_optimal_divide(mask,memoryFixPerVoxel,memoryDynamicPerVoxel);

            % parameter estimation
            out = [];
            for ks = 1:NSegment
                
                if NSegment ~= 1
                    fprintf('Running #Segment = %d/%d \n',ks,NSegment);
                    disp   ('------------------------')
                end
    
                % determine slice# given a segment
                if ks ~= NSegment
                    slice = 1+(ks-1)*maxSlice : ks*maxSlice;
                else
                    slice = 1+(ks-1)*maxSlice : dims(3);
                end
                
                % divide the data
                dwi_tmp     = dwi(:,:,slice,:);
                mask_tmp    = mask(:,:,slice);
                if ~isempty(pars0); for km = 1:numel(this.modelParams); pars0_tmp.(this.modelParams{km}) = pars0.(this.modelParams{km})(:,:,slice); end
                else;               pars0_tmp = [];                 end
                fields      = fieldnames(extraData); for kfield = 1:numel(fields); if all(size(extraData.(fields{kfield}),1:3) == size(mask)); extraData_tmp.(fields{kfield}) = extraData.(fields{kfield})(:,:,slice,:,:,:,:,:,:,:,:); end; end

                % run fitting
                [out_tmp]    = this.fit(dwi_tmp,mask_tmp,fitting,extraData_tmp,pars0_tmp);

                % restore 'out' structure from segment
                out = utils.restore_segment_structure(out,out_tmp,slice,ks);

            end
            out.mask = mask;
            %%%%%%%%%%%%%%%% End Step 2 %%%%%%%%%%%%%%%%

            % save the estimation results if the output filename is provided
            askadam.save_askadam_output(fitting.outputFilename,out)

        end

        % Data fitting function, can be 2D (voxel-based) or 4D (image-based)
        function [out] = fit(this,dwi,mask,fitting,extraData,pars0)
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
        % fa        : final Intraneurite volume fraction
        % Da        : final Intraneurite diffusivity (um2/ms)
        % De        : final Extraneurite diffusivity (um2/ms)
        % ra        : final exchange rate from intra- to extra-neurite compartment
        % p2        : final dispersion index (if fitting.lax=2)
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
            if isempty(pars0);  pars0 = this.determine_x0(dwi,mask,fitting,extraData); end

            % You may add more dispay messages here
            disp('---------------------------');
            disp('Additional model parameters');
            disp('---------------------------');
            disp(['NEXI Rice rotational invariant model, lmax = ' num2str(fitting.lmax)]);
            disp('---------------------------');

            % 2.3 askAdam optimisation main
            askadamObj = askadam();

            % askadam.optimisation does not see extractData so we have to manually mask the data inside here, make sure the voxel is on the second dimension
            % extraData   = structfun(@transpose, utils.gpu_vectorise_NDto2D_struct(extraData,mask) ,'UniformOutput',false);
            extraData   = utils.masking_ND2GD_preserve_struct(extraData,mask) ;

            % initiate starting points arrays
            out = askadamObj.optimisation( dwi, mask, w, pars0, fitting, @this.FWD, extraData);
            % out = askadamObj.optimisation( dwi, mask, w, pars0, fitting, @this.FWD, fitting.lmax);

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
        function [dwi,mask,extradata] = prepare_dwi_data(this,dwi,mask,extradata,lmax)
            % full DWI data then compute rotaionally invariant signal

            dims = size(dwi,1:3);

            % normalised sigma by b0
            ind_b0 = extradata.bval == 0;
            if sum(ind_b0==0)==numel(ind_b0)
                dwi_b0      = ones(dims);
            else
                dwi_b0      = mean(dwi(:,:,:,ind_b0),4);
            end
            extradata.sigma = extradata.sigma ./ dwi_b0;
            extradata.sigma = max(utils.set_nan_inf_zero(extradata.sigma),askadam.epsilon);

            if size(dwi,4)/(lmax/2+1) > numel(this.b) 
                % compute spherical mean signal
                fprintf('Computing rotationally invariant signal...')

                % if the inout little delta is one value then create a vector
                if numel(extradata.ldelta) == 1
                    extradata.ldelta = ones(size(extradata.bval)) * extradata.ldelta;
                end
                DWIutilityObj = DWIutility();
                [dwi]   = DWIutilityObj.get_Sl_all(dwi,extradata.bval,extradata.bvec,extradata.ldelta,extradata.BDELTA,lmax);

                fprintf('done.\n');

            elseif size(dwi,4) < numel(this.b)
                error('There are more b-shells in the class object than available in the input data. Please check your input data.');
            end
            
            % make sure no NaN/Inf in signal
            [dwi, mask_naninf] = utils.set_nan_inf_zero(dwi); mask_naninf = max(mask_naninf,[],4);
            mask_valid         = and(and(mask, max(dwi,[],4)<=1.01),~mask_naninf);
            % make sure not fitting SNR too low
            mask_LowSNR = extradata.sigma > 1;  % SNR = 1
            mask_valid  = and(mask_valid,~mask_LowSNR);
            
            % check signal similarity
            dwi_2D          = utils.reshape_ND2GD(dwi,mask_valid);
            signalTemplate  = mean(dwi_2D,2,"omitmissing");         % assuming the majority of the signal are from tissues
            signalTemplate  = (signalTemplate - mean(signalTemplate)) ./ std(signalTemplate);
            Rcorr           = zeros(1,size(dwi_2D,2));
            for k = 1:size(dwi_2D,2)
                signalVoxel = dwi_2D(:,k);
                signalVoxel = (signalVoxel - mean(signalVoxel)) ./ std(signalVoxel);
                Rcorr(k)    = corr(signalTemplate,signalVoxel);
            end
            Rcorr            = utils.reshape_GD2ND(Rcorr,mask_valid);
            mask_dissimilar  = Rcorr < 0.1;
            mask_valid       = and(mask_valid,~mask_dissimilar);
            
            if numel(mask_valid(mask_valid)) ~= numel(mask(mask)) 
                disp('Signal mask is updated! Please use the fitted mask in your subsequent analysis.');
                mask = mask_valid;
            end

            % thres_SNR               = 20;
            % extradata.maskhighSNR   = (dwi ./extradata.sigma) > thres_SNR;

        end

        %%%%% Prior estimation related functions %%%%%

        % determine how the starting points will be set up
        function x0 = determine_x0(this,y,mask,fitting,extradata) 

            disp('---------------');
            disp('Starting points');
            disp('---------------');

            dims = size(mask,1:3);

            if ischar(fitting.start)
                switch lower(fitting.start)
                    case 'likelihood'
                        % using maximum likelihood method to estimate starting points
                        sigma = median(extradata.sigma(mask>0));
                        x0 = this.estimate_prior(y,mask,[],sigma);
                        % x0 = this.estimate_prior(y,mask,[],extradata);
    
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
        function pars0 = estimate_prior(this,dwi,mask, Nsample,sigma)
        % Estimation starting points for NEXI using likehood method

            start = tic;
            
            disp('Estimate starting points based on likelihood ...')

            lmax = 0;

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

            if nargin < 4 || isempty(Nsample)
                Nsample         = 1e4;
            end
            % create training data
            [x_train, S_train] = this.traindata(Nsample);
            S_train = this.myriceM1(S_train,sigma);

            % reshape input data,  put DWI dimension to 1st dim
            % dims    = size(dwi);
            dwi     = utils.reshape_ND2GD(dwi,mask);
            % sigma   = utils.reshape_ND2GD(extradata.sigma,mask);
            % dwi     = permute(dwi,[4 1 2 3]);
            % dwi     = reshape(dwi,[dims(4), prod(dims(1:3))]);

            % find masked voxels
            % ind         = find(mask(:));
            if lmax == 0
                Nparam = 4;
            elseif lmax == 2
                Nparam = 5;
            end

            % pars0_mask  = zeros(Nparam,length(ind));
            pars0_mask  = zeros(Nparam,size(dwi,2));
            if ~isempty(pool)
                parfor kvol = 1:size(dwi,2)
                    pars0_mask(:,kvol) = this.likelihood(dwi(:,kvol), x_train, S_train);
                end
            else
                for kvol = 1:size(dwi,2)
                    pars0_mask(:,kvol) = this.likelihood(dwi(:,kvol), x_train, S_train);
                end
            end
            pars = utils.reshape_GD2ND(pars0_mask,mask);        
            % pars           = zeros(Nparam,size(dwi,2));
            % pars(:,ind)    = pars0_mask;
            % 
            % % reshape estimation into image
            % pars           = permute(reshape(pars,[size(pars,1) dims(1:3)]),[2 3 4 1]);

            % Correction for CSF
            bval_thres      = max(min(gather(this.b)),1.1);
            idx             = gather(this.b) <= bval_thres;
            D0              = real(this.b(idx)\-log(dwi(cat(1,idx,false(size(idx))),:)));
            D0              = utils.reshape_GD2ND(D0,mask);
            % D0              = permute(reshape(D0,[size(D0,1) dims(1:3)]),[2 3 4 1]);
            D0              = max(utils.set_nan_inf_zero(D0),1e-8);
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

        end

        % create training data for likelihood
        function [x_train, S_train, intervals] = traindata(this, N_samples, varargin)
            if nargin < 4
                intervals = [0.01 0.99  ;   % fa
                              1.5 3     ;   % Da
                              0.5 1.5   ;   % De
                                1 100   ;   % exchange time = (1-fa)/r
                             0.01 0.99 ];   % p2
            else
                intervals = varargin{1};
            end
            lmax = 0;
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
                    % % Rician floor correction
                    % Sl0(:,j) = this.myriceM1(Sl0(:,j), sigma);
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
        function [pars_best, sse_best] = likelihood(this, S0, x_train, S_train)
            lmax = 0;
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

        %% NEXI signal related functions

        % Forward model to generate NEXI signal
        function [s] = FWD(this, pars, extraData)
        
            fa   = max(pars.fa, askadam.epsilon); % avoid division by zeros when computing re
            Da   = pars.Da;
            De   = pars.De;
            ra   = pars.ra;
    
            % Sl0
            s = this.Sl0(fa, Da, De, ra);
            
            % % Sl2
            % if lmax == 2
            %     p2  = pars.p2;
            %     s   = cat(1,s,this.Sl2(fa, Da, De, ra, p2));
            % end

            % Rician floor correction
            s = this.myriceM1(s, extraData.sigma);
            % thres_SNR       = 20;
            % maskLowSNR      = s./extraData.sigma <= thres_SNR;
            % sRice           = this.myriceM1(s, extraData.sigma);
            % s(maskLowSNR)   = sRice(maskLowSNR);


            % make sure s cannot be greater than 1
            s = min(s,1);   % s = [Nb, Nvoxel]
                
        end
        
        % 0th order rotational invariant
        function S = Sl0(this, fa, Da, De, ra)

            if isgpuarray(fa)
                bval    = gpuArray(single(this.b));
                DELTA   = gpuArray(single(this.Delta));
            else
                bval = this.b;
                DELTA   = this.Delta;
            end
            
            Da = bval.*Da;
            De = bval.*De;
            ra = DELTA.*ra;
            re = ra.*fa./max(1-fa,askadam.epsilon);
            
            % Trapezoidal's rule replacement
            Nx  = 14;    % NRMSE<0.05% for Nx=14
            x   = zeros([ones(1,ndims(fa)), Nx],'like',bval); x(:) = linspace(0,1,Nx);
            S   = trapz(x(:),this.M(x, fa, Da, De, ra, re),ndims(x));
            % myfun = @(x) this.M(x, fa, Da, De, ra, re);
            % S = integral(myfun, 0, 1, 'AbsTol', 1e-14, 'ArrayValued', true);

        end
        
        % 2nd order rotational invariant
        function S = Sl2(this, fa, Da, De, ra, p2)

            if isgpuarray(fa)
                bval    = gpuArray(single(this.b));
                DELTA   = gpuArray(single(this.Delta));
            else
                bval = this.b;
                DELTA   = this.Delta;
            end

            Da = bval.*Da;
            De = bval.*De;
            ra = DELTA.*ra;
            re = ra.*fa./(1-fa);
            
            % Trapezoidal's rule replacement
            Nx  = 14;    % NRMSE<0.5% for Nx=14
            x   = zeros([ones(1,ndims(fa)), Nx],'like',bval); x(:) = linspace(0,1,Nx);
            S   = trapz(x(:),this.M(x, fa, Da, De, ra, re).*(3*x.^2-1)/2,ndims(x));
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

        function M1 = myriceM1( nu, sigma)
            % if sigma < 1/75
                % TODO: this neeeded to be tested
                % M1 = sigma .* sqrt(pi/2) .* hypergeom(-1/2, 1, -nu.^2/2./sigma.^2);
                % disp('SNR is too high. PLease use the ordinary NEXI model instead.')
            % else
                M1              = max(sigma .* sqrt(pi/2) .* gpuNEXIrice.myL12(-nu.^2/2./sigma.^2),nu);
                % M1              = min(M1,max(nu));
                % M1(isnan(M1))   = nu(isnan(M1));
                % M1(isinf(M1))   = nu(isinf(M1));
            % end
        end

        function y = myL12(x)
            x = max(x,-170);
            % TODO: besseli does not support dlarray
            y = exp(x/2) .* ( (1-x).*gpuNEXIrice.mybesseli(0,-x/2) - x.*gpuNEXIrice.mybesseli(1,-x/2) );
            % y = exp(x/2) .* ( (1-x).*besseli(0,-x/2) - x.*besseli(1,-x/2) );
            % y = hypergeom(-1/2,1,x);
        end

        function y = L12(x)
            y = exp(x/2) .* ( (1-x).*besseli(0,-x/2) - x.*besseli(1,-x/2) );
        end

        function I = mybesseli(nu,z)
            Nx  = 34;    % NRMSE<0.05% for Nx=34
            x   = zeros([ones(1,ndims(z)), Nx]); x(:) = linspace(0,pi,Nx);
            I   = 1/pi * trapz(x(:),exp(z.*cos(x)).*cos(nu*x),ndims(x));
        end

        %% Utilities
        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default(fitting)
            % get basic fitting setting check
            fitting2 = askadam.check_set_default_basic(fitting);

            % get customised fitting setting check
            if ~isfield(fitting,'regmap');          fitting2.regmap     = 'fa';             end
            if ~isfield(fitting,'lmax');            fitting2.lmax       = 0;                end
            if ~isfield(fitting,'start');           fitting2.start      = 'likelihood';     end

            if ~iscell(fitting2.regmap)
                fitting2.regmap = cellstr(fitting2.regmap);
            end

        end

    end

end