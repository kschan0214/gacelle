classdef gpumcmicro < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 29 September 2025
% Date modified: 13 November 2025

% TODO: sort b and te and input full DWI

    properties
        % default model parameters and estimation boundary
        % f     : neurite fraction (f=fa/(fa+fe)), 
        % D     : intrinsic diffusivity [um2/ms]
        % R2a   : Intra-neurite R2 [1/s] (for multi-echo)
        % R2e   : Extra-neurite R2 [1/s] (for multi-echo)
        modelParams     = { 'f';  'D';'R2a';'R2e';'noise'};
        ub              = [   1;    3;   50;   75; 0.1];
        lb              = [1e-6; 1e-6;    1;    1; 0.001];
        startPoint      = [ 0.6;    2;   12;   30; 0.005];
        step            = [0.05; 0.04; 2.58; 2.58; 0.005];

    end

    properties (GetAccess = public, SetAccess = protected)
        b;
        te;
    end
    
    properties (GetAccess = private, SetAccess = private)
    end
    
    methods

        % constructuor
        function this = gpumcmicro(b, te)
        % Estimation of neurite fraction and intrinsic diffusivity using SMT with the option of performing compartmental T2 mapping
        % GACELLE's implementation of Kaden E, Kelm ND, Carson RP, Does MD, and Alexander DC: Multi-compartment microscopic diffusion imaging. NeuroImage, vol. 139, pp. 346–359, 2016. DOI: 10.1016/j.neuroimage.2016.06.002
        % smt = gpumcmicro(b, te)
        %       output:
        %           - smt: object of a fitting class
        %
        %       input:
        %           - b: b-value [ms/um2]
        %           - te: echo time [s] (optional)
        %
        %  Authors: 
        %  Kwok-Shing Chan (kchan2@mgh.harvard.edu)
        %
            
            this.b      = (single( b(:)) );
            % relaxation
            if nargin<2 || isempty(te)
                this.te = zeros(size(this.b));    % no input te assume single TE
            else
                if isscalar(unique(te))
                    this.te = zeros(size(this.b),'like',this.b);    % only 1 TE then set TE to 0
                else
                    this.te = (single(te(:)));    % same length as b or scalar (if Nt==1)
                end
            end

        end

        % update properties according to lmax
        function this = updateProperty(this, fitting)

            % only 1 TE
            if isscalar(unique(this.te))
                for kpar = {'R2a','R2e'}
                    idx = find(ismember(this.modelParams,kpar));
                    this.modelParams(idx)   = [];
                    this.lb(idx)            = [];
                    this.ub(idx)            = [];
                    this.startPoint(idx)    = [];
                    this.step(idx)          = [];
                end
            end

            % whether fitting D or not
            if ~fitting.isFitD
                idx = find(ismember(this.modelParams,'D'));
                this.modelParams(idx)       = [];
                this.lb(idx)                = [];
                this.ub(idx)                = [];
                this.startPoint(idx)        = [];
                this.step(idx)              = [];
            end

            % property change in related to solver
            if ~strcmpi(fitting.solver,'mcmc')
                idx = find(ismember(this.modelParams,'noise'));
                this.modelParams(idx)       = [];
                this.lb(idx)                = [];
                this.ub(idx)                = [];
                this.startPoint(idx)        = [];
                this.step(idx)              = [];
            else
                this.b  = gpuArray(single(this.b));
                this.te = gpuArray(single(this.te));
            end

        end
    
        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('=============================================');
            disp('Multi-compartment microscopic diffusion model');
            disp('=============================================');

            disp('----------------')
            disp('Data Information');
            disp('----------------')
            fprintf('b-shells (ms/um2)              : [%s] \n',num2str(this.b.',' %.2f'));
            if ~isscalar(unique(this.te))
                fprintf('TE (ms)                        : [%s] \n',num2str(this.te.'*1e3,' %.2f'));
            end

        end

        %% higher-level data fitting functions

        % This is a wrapper of the 'fit' function.
        % The main purpose of this function is to handle memory issue and ensure the input data is correct for 'fit'
        function  [out] = estimate(this, dwi, mask, fitting, extraData, pars0)
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
        % fitting   : fitting algorithm parameters (see fit function)
        % 
        % Output
        % -----------
        % out       : output structure contains all parameter estimation results
        % 
            
            % display basic info
            this.display_data_model_info;

            % get all fitting algorithm parameters 
            fitting = this.check_set_default(fitting);

            % get matrix size
            dims = size(dwi,1:3);

            %%%%%%%%%%%%%%%% Step 1: Validate all input data %%%%%%%%%%%%%%%%
            % if no pars input at all (not even empty) then use prior
            if nargin < 6; pars0        = []; end
            if nargin < 5; extraData    = []; end

            % compute rotationally invariant signal if needed
            [this,dwi] = this.prepare_dwi_data(dwi,extraData,0);

            % mask sure no nan or inf in data
            [dwi,mask] = utils.remove_img_naninf(dwi,mask);

            % convert datatype to single or logical
            dwi     = single(dwi);
            mask    = mask >0;
            if ~isempty(pars0); for km = 1:numel(this.modelParams); pars0.(this.modelParams{km}) = single(pars0.(this.modelParams{km})); end; end

            %%%%%%%%%%%%%%%% End Step 1 %%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%% Step 2: Validate if GPU has enough memory  %%%%%%%%%%%%%%%%
            % TODO: memory management
            % determine if we need to divide the data to fit in GPU
            % gpool = gpuDevice;  reset(gpool);
            % memoryFixPerVoxel       = 0;   % get this number based on mdl fit
            % memoryDynamicPerVoxel   = 0;     % get this number based on mdl fit
            % [NSegment,maxSlice]     = utils.find_optimal_divide(mask,memoryFixPerVoxel,memoryDynamicPerVoxel);
            
            % % parameter estimation
            % out = [];
            % for ks = 1:NSegment
            % 
            %     fprintf('Running #Segment = %d/%d \n',ks,NSegment);
            %     disp   ('------------------------')
            % 
            %     if ks ~= NSegment
            %         slice = 1+(ks-1)*maxSlice : ks*maxSlice;
            %     else
            %         slice = 1+(ks-1)*maxSlice : dims(3);
            %     end
            % 
            %     dwi_tmp     = dwi(:,:,slice,:);
            %     mask_tmp    = mask(:,:,slice);
            %     if ~isempty(pars0); for km = 1:numel(this.modelParams); pars0_tmp.(this.modelParams{km}) = pars0.(this.modelParams{km})(:,:,slice); end
            %     else;                                                    pars0_tmp = [];                 end
            %     if ~isempty(extraData); fields      = fieldnames(extraData); for kfield = 1:numel(fields); extraData_tmp.(fields{kfield}) = extraData.(fields{kfield})(:,:,slice,:,:,:,:,:,:,:,:); end
            %     else;                                                    extraData_tmp = [];                 end
            % 
            %     [out_tmp]  = this.fit(dwi_tmp,mask_tmp,fitting,extraData_tmp,pars0_tmp);
            % 
            %     % restore 'out' structure from segment
            %     out = utils.restore_segment_structure(out,out_tmp,slice,ks);
            % 
            % end
            [out]  = this.fit(dwi,mask,fitting,extraData,pars0);
            out.mask = mask;
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
        function [out] = fit(this,dwi,mask,fitting, extraData, pars0)
        %
        % Input
        % -----------
        % dwi       : S0 normalised 4D dwi images, [x,y,slice,diffusion], 4th dimension corresponding to [Sl0_b1,Sl0_b2 etc.]; the order of bval must match the order in the constructor 
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
        % pars0     : structure variable of starting points of fitting (optional)
        % 
        % Output
        % -----------
        % out       : output structure
        %   .final      : final results
        %   .min        : results with the minimum loss metric across all iterations
        %       .loss       : loss metric      
        %
        % Kwok-Shing Chan @ MGH
        % kchan2@mgh.harvard.edu
        % Date created: 8 Dec 2023
        % Date modified:
        %
            
            % check GPU
            gpool = gpuDevice;
            
            % check image size
            dims = size(dwi,1:3);

            %%%%%%%%%%%%%%%%%%%% Step 1. Validate and parse input %%%%%%%%%%%%%%%%%%%%
            if nargin < 3 || isempty(mask); mask = ones(dims,'logical'); end % if no mask input then fit everthing
            if nargin < 4; fitting = struct(); end
            if nargin < 5; extraData = []; end
            % set initial tarting points
            if nargin < 6; pars0 = []; % no initial starting points
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
            w = this.compute_optimisation_weights(mask,fitting.lossFunction,0); % This is a customised funtion

            % 2.2 estimate prior if neede
            if isempty(pars0);  pars0 = this.determine_x0(dwi,mask,fitting); end

            if ~isempty(extraData); extraData   = utils.masking_ND2GD_preserve_struct(extraData,mask) ; end

            % 2.3 askAdam optimisation main
            switch fitting.solver
                case 'askadam'
                    askadamObj  = askadam(); 
                    out         = askadamObj.optimisation(dwi, mask, w, pars0, fitting, @this.FWD, extraData,fitting.solver);
                case 'mcmc'
                    mcmcObj     = mcmc(); 
                    out         = mcmcObj.optimisation(dwi, mask, w, pars0, fitting, @this.FWD, extraData,fitting.solver);
            end
            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            disp('The estimation is completed.');
            
            % clear GPU
            reset(gpool)

        end

        %% Data preparation

        % compute rotationally invariant DWI signal if necessary
        function [this,dwi] = prepare_dwi_data(this,dwi,extradata,lmax)
            % full DWI data then compute rotaionally invariant signal
            if size(dwi,4)/(lmax/2+1) > numel(this.b) 

                % compute rotationally invariant signal
                if ~isfield(extradata,'te')
                    extradata.te = zeros(size(extradata.bval));
                end
                DWIutils                        = DWIutility();
                [dwi,bval_sorted,~,~,te_sorted] = DWIutils.compute_rotationally_invariant_signal(dwi,extradata.bval,extradata.bvec,[],[],extradata.te,lmax);
                
                % update b and te order
                this.b  = single(bval_sorted);
                this.te = single(te_sorted);

            elseif size(dwi,4) < numel(this.b)
                error('There are more b-shells in the class object than available in the input data. Please check your input data.');
            end

            % normalised by the first volume
            dwi = dwi ./ dwi(:,:,:,1);
        end

        % compute weights for optimisation
        function w = compute_optimisation_weights(this,mask,lossFunction,lmax)
        % 
        % Output
        % ------
        % w         : N-D signal masked wegiths
        %
            dims = size(mask,1:3);

            % lmax dependent weights
            l = 0:2:lmax;
            w = zeros([dims numel(this.b)*numel(l)],'single');
            for kl = 1:(lmax/2+1)
                for kb = 1:numel(this.b)
                    w(:,:,:,(kl-1)*numel(this.b)+kb) = 1/ (2*l(kl)+1);
                end
            end
            % if L1 then take square root
            if strcmpi(lossFunction,'l1')
                w = sqrt(w);
            end
            w = w ./ max(w(:));
            
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
                        x0 = this.estimate_prior(y,mask,[]);
    
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
            x0 = mcmc.set_boundary(x0,fitting.ub,fitting.lb);
            
            fprintf('Estimation lower bound [%s]: [%s]\n',      cell2str(this.modelParams),replace(num2str(fitting.lb(:).',' %.2f'),' ',','));
            fprintf('Estimation upper bound [%s]: [%s]\n',      cell2str(this.modelParams),replace(num2str(fitting.ub(:).',' %.2f'),'  ',','));
            ('---------------');
        end

        % using maximum likelihood method to estimate starting points
        function pars0 = estimate_prior(this,dwi,mask, Nsample)
        % Estimation starting points for NEXI using likehood method

            start = tic;
            
            disp('Estimate starting points based on likelihood ...')

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

            % reshape input data,  put DWI dimension to 1st dim
            dims    = size(dwi);
            dwi     = permute(dwi,[4 1 2 3]);
            dwi     = reshape(dwi,[dims(4), prod(dims(1:3))]);

            % find masked voxels
            ind         = find(mask(:));

            Nparam = size(x_train,1);

            pars0_mask  = zeros(Nparam,length(ind),'single');
            if ~isempty(pool)
                parfor kvol = 1:length(ind)
                    pars0_mask(:,kvol) = this.likelihood(dwi(:,ind(kvol)), x_train, S_train);
                end
            else
                for kvol = 1:length(ind)
                    pars0_mask(:,kvol) = this.likelihood(dwi(:,ind(kvol)), x_train, S_train);
                end
            end
            pars           = zeros(Nparam,size(dwi,2),'single');
            pars(:,ind)    = pars0_mask;

            % reshape estimation into image
            pars           = permute(reshape(pars,[size(pars,1) dims(1:3)]),[2 3 4 1]);

            ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
            fprintf('Starting points estimated. Elapsed time (hh:mm:ss): %s \n',string(ET));
            if isDeletepool
                delete(pool);
            end

            for km = 1:size(pars,4)
                pars0.(this.modelParams{km}) = pars(:,:,:,km); ...
            end

            % noise for mcmc
            if ~isempty(find(ismember(this.modelParams,'noise'), 1))
                pars0.(this.modelParams{end}) = single(ones(size(mask)) * this.startPoint(end));
            end

        end

        % generate training data for likelihood
        function [x_train, S_train, intervals] = traindata(this, N_samples, varargin)
            if nargin < 3
                intervals = [0.1 0.9      ;   % neurite fraction
                             1.7 2.9      ;   % intrinsic diffusivity
                               7 15     ;   % neurite R2
                              20 50]   ;   % extra-cellular R2
            else
                intervals = varargin{1};
            end

            if isempty(find(ismember(this.modelParams,'noise'), 1))
                numParam    = numel(this.modelParams);
            else
                numParam    = numel(this.modelParams) - 1;
            end
            numBSample  = numel(gather(this.b));
            
            % batch size can be modified according to available hardware
            extraData   = [];
            batch_size  = 1e3;
            reps        = ceil(N_samples/batch_size);
            x_train     = zeros(numParam,batch_size,reps,'single');
            S_train     = zeros(numBSample,batch_size,reps,'single');
            for k = 1:reps
                % generate random parameter guesses and construct batch for NN signal evaluation
                pars = intervals(:,1) + diff(intervals,[],2).*rand(size(intervals,1),batch_size);

                params.f    = pars(1,:);
                if any(ismember(this.modelParams,'R2a'))
                    params.R2a  = pars(3,:);
                    params.R2e  = pars(4,:);
                else
                    pars(4,:) = [];
                    pars(3,:) = [];
                end
                if any(ismember(this.modelParams,'D'))
                    params.D    = pars(2,:);
                else
                    extraData.D = pars(2,:);
                    pars(2,:)   = [];
                end

                S = this.FWD(params,extraData);

                % remaining signals (dot, soma)
                x_train(:,:,k) = pars;
                S_train(:,:,k) = S;

            end

        end
        
        % likelihood
        function [pars_best, sse_best] = likelihood(this, S0, x_train, S_train)
            
            lmax = 0;
            wt = kron(ones(size(this.b)), 1./(2*(0:2:lmax)+1));
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
    
        %%  Signal related functions
        
        % Forward model
        function s = FWD(this, pars, extraData, solver)

            if nargin<4
                solver = [];
            end

            f   = pars.f;
            if isfield(pars,'D')
                D   = pars.D;
            else
                D = extraData.D;
            end
            if isfield(pars,'R2a')
                R2a = pars.R2a;
                R2e = pars.R2e;
            else
                R2a = 0;
                R2e = 0;
            end

            % Forward model
            if strcmpi(solver,'mcmc')
                s = arrayfun(@diffusion_relaxation_spherical_mean_combine,f, D, R2a, R2e, this.b,this.te);
            else
                s = this.diffusion_relaxation_spherical_mean_combine(f, D, R2a, R2e, this.b,this.te);
            end
            % normalised to 1st echo, b=0
            s = s ./ s(1,:,:,:,:);
                
            % make sure s cannot be greater than 1
            s = min(s,1);
                
        end


    end

    methods(Static)

        %% Utility

        function s = diffusion_relaxation_spherical_mean_combine(f, D, R2a, R2e, b,te)

            % avoid division by zeros
            b = max(b,1e-10);
            
            % 1st order tortuosity approximation
            Dr = (1-f).*D;
            
            % axonal stick compartment
            Sa = sqrt(pi./(4*(b.*D))) .* erf(sqrt(b.*D)) .* exp(-te.*R2a);
            % extraceullular axi-symmetric compartment
            Se = sqrt(pi./(4.*(D - Dr).*b)) .* exp(-b.*Dr) .* erf(sqrt(b .*(D - Dr))) .* exp(-te.*R2e);
            % combining signal
            s = f.*Sa + (1-f).*Se;
        
        end

        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default(fitting)
            % get basic fitting setting check
            if ~isfield(fitting,'solver');      fitting.solver = 'askadam';        end
            if strcmpi(fitting.solver,'askadam')
                fitting2 = askadam.check_set_default_basic(fitting);

                if ~isfield(fitting,'regmap');      fitting2.regmap = {'f'};            end

                if ~iscell(fitting2.regmap)
                    fitting2.regmap = cellstr(fitting2.regmap);
                end

            else
                fitting2 = mcmc.check_set_default_basic(fitting);
                fitting2.lossFunction = [];
            end

            % get customised fitting setting check
            if ~isfield(fitting,'start');       fitting2.start  = 'likelihood';     end
            if ~isfield(fitting,'isFitD');      fitting2.isFitD = true;             end
            
        end
    
    end

end