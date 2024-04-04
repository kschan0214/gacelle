classdef gpuMODEL
% This is the skeleton script for new model with askAdam solver
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 3 April 2024
% Date modified: 

    properties
        % default model parameters and estimation boundary
        model_params    = {'fa','Da','De','ra','p2'};
        ub              = [   1,   3,   3,   1,  1];
        lb              = [ eps, eps, eps,1/250, eps];
    end

    properties (GetAccess = public, SetAccess = protected)
        b;
        Delta;  
        Nav;
        askadamObj;
    end
    
    methods

        % constructuor
        function this = gpuMODEL(b, Delta, varargin)
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
            
            this.askadamObj = askadam();
            this.b      = b(:) ;
            this.Delta  = Delta(:) ;
            if nargin > 2
                this.Nav = varargin{1} ;
            else
                this.Nav =  ones(size(b)) ;
            end
            this.Nav = this.Nav(:) ;
        end
        
        %% higher-level data fitting functions
        % Wrapper function of fit to handle image data; automatically segment data and fitting in case the data cannot fit in the GPU in one go
        function  [out, fa, Da, De, ra, p2] = estimate(this, dwi, mask, extradata, fitting, pars0)
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
            
            disp('========================');
            disp('NEXI with askAdam solver');
            disp('========================');

            disp('----------------')
            disp('Data Information');
            disp('----------------')
            fprintf('b-shells (ms/um2)              : [%s] \n',num2str(this.b.',' %.2f'));
            fprintf('Diffusion time (ms)            : [%s] \n\n',num2str(this.Delta.',' %i'));

            % get all fitting algorithm parameters 
            fitting = this.check_set_default(fitting);

            % compute rotationally invariant signal if needed
            dwi = this.prepare_dwi_data(dwi,extradata,fitting.lmax);

            % mask sure no nan or inf
            [dwi,mask] = this.askadamObj.remove_img_naninf(dwi,mask);

            % get matrix size
            dims = size(dwi);

            % check prior input
            if nargin < 6
                pars0 = [];
            end
            if and(fitting.isPrior,isempty(pars0))
                pars0 = this.estimate_prior(dwi, mask, [], fitting.lmax);
            end

            % convert datatype to single
            dwi     = single(dwi);
            mask    = mask >0;
            if ~isempty(pars0); pars0 = single(pars0); end

            % determine if we need to divide the data to fit in GPU
            g = gpuDevice; reset(g);
            memoryFixPerVoxel       = 0.0013;   % get this number based on mdl fit
            memoryDynamicPerVoxel   = 0.05;     % get this number based on mdl fit
            [NSegment,maxSlice]     = this.askadamObj.find_optimal_divide(mask,memoryFixPerVoxel,memoryDynamicPerVoxel);

            % parameter estimation
            out = [];
            fa = zeros(dims(1:3),'single');
            Da = zeros(dims(1:3),'single');
            De = zeros(dims(1:3),'single');
            ra = zeros(dims(1:3),'single');
            p2 = zeros(dims(1:3),'single');
            for ks = 1:NSegment

                fprintf('Running #Segment = %d/%d \n',ks,NSegment);
                disp   ('------------------------')
    
                % determine slice# given a segment
                if ks ~= NSegment
                    slice = 1+(ks-1)*maxSlice : ks*maxSlice;
                else
                    slice = 1+(ks-1)*maxSlice : dims(3);
                end
                
                % divide the data
                dwi_tmp     = dwi(:,:,slice,:);
                mask_tmp    = mask(:,:,slice);
                if ~isempty(pars0); pars0_tmp = pars0(:,:,slice,:);
                else;               pars0_tmp = [];                 end

                % run fitting
                if fitting.lmax >0
                    [out_tmp, fa(:,:,slice), Da(:,:,slice), De(:,:,slice), ra(:,:,slice), p2(:,:,slice)]    = this.fit(dwi_tmp,mask_tmp,fitting,pars0_tmp);
                else
                    [out_tmp, fa(:,:,slice), Da(:,:,slice), De(:,:,slice), ra(:,:,slice)]                   = this.fit(dwi_tmp,mask_tmp,fitting,pars0_tmp);
                    p2 = [];
                end

                % restore 'out' structure from segment
                out = this.askadamObj.restore_segment_structure(out,out_tmp,slice,ks);

            end
            out.mask = mask;

            % save the estimation results if the output filename is provided
            this.askadamObj.save_askadam_output(fitting.output_filename,out)

        end

        % Data fitting function, can be 2D (voxel-based) or 4D (image-based)
        function [out, fa, Da, De, ra, p2] = fit(this,dwi,mask,fitting,pars0)
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
            g = gpuDevice;
            
            % get image size
            dims = size(dwi);

            %%%%%%%%%%%%%%%%%%%% 1. Validate and parse input %%%%%%%%%%%%%%%%%%%%
            if nargin < 3 || isempty(mask)
                % if no mask input then fit everthing
                mask = ones(dims);
            else
                % if mask is 3D
                if ndims(mask) < 4
                    mask = repmat(mask,[1 1 1 dims(4)]);
                end
            end

            if nargin < 4
                fitting = struct();
            end

            % set initial tarting points
            if nargin < 5
                % no initial starting points
                pars0 = [];
            else
                pars0 = single(pars0);
            end

            % get all fitting algorithm parameters 
            fitting         = this.check_set_default(fitting);
            fitting.Nsample = numel(mask(mask ~= 0)) / dims(4); 

            % determine how many model parameters
            if      fitting.lmax == 0;  fitting.model_params = this.model_params(1:4);
            elseif  fitting.lmax == 2;  fitting.model_params = this.model_params;     end

            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(fitting.model_params)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(fitting.model_params)); end

            % put data input gpuArray
            mask = gpuArray(logical(mask));  
            dwi  = gpuArray(single(dwi));
            
            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w = this.compute_optimisation_weights(mask,dims,fitting.lossFunction,fitting.lmax); % This is a customised funtion
            w = dlarray(gpuArray(w).','CB');

            % 2.2 display optimisation algorithm parameters
            this.askadamObj.display_basic_fitting_parameters(fitting);
            % You may add more dispay messages here
            disp(['lmax                     = ' num2str(fitting.lmax)]);
            
            % 2.3 askAdam optimisation main
            out = this.askadamObj.optimisation(pars0, @this.modelGradients, dwi, mask, w, fitting);

            % 2.4 organise output
            p2 = [];
            fa = out.final.fa;
            Da = out.final.Da;
            De = out.final.De;
            ra = out.final.ra;
            if fitting.lmax == 2
                p2 = out.final.p2;
            end

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            disp('The process is completed.')
            
            % clear GPU
            reset(g)
            
        end

        % compute the gradient and loss of forward modelling
        function [gradients,loss,loss_fidelity,loss_reg] = modelGradients(this, parameters, data, mask, weights, fitting)
        % Designed your model gradient function here
        %
        % For the first 6 input (this -> fitting), you MUST following the same order as above, any newly introduced variables needed to be put after fitting
        %
        % Input
        % -----
        % parameters    : structure contains all model parameters
        % data          : N-D measured data
        % mask          : N-D signal mask, same size as data
        % weights       : 1D signal weights (already masked)
        % fitting       : fitting algorithm parameters
        % 
            % rescale network parameter to true values
            parameters = this.askadamObj.rescale_parameters(parameters,fitting.lb,fitting.ub,fitting.model_params);
            
            mask    = permute(mask,[4 1 2 3]);
            data    = permute(data,[4 1 2 3]);
            for k = 1:numel(fitting.ub)
                parameters.(fitting.model_params{k}) = permute(parameters.(fitting.model_params{k}),[4 1 2 3]);
            end

            % Forward model
            % R           = this.FWD(parameters,fitting);
            R           = this.FWD(parameters,fitting,mask(1,:,:,:));
            R(isinf(R)) = 0;
            R(isnan(R)) = 0;

            % Masking
            % R   = dlarray(R(mask>0).',     'CB');
            R       = dlarray(R(:).',           'CB');
            data    = dlarray(data(mask>0).',    'CB');

            % Data fidelity term
            switch lower(fitting.lossFunction)
                case 'l1'
                    loss_fidelity = l1loss(R, data, weights);
                case 'l2'
                    loss_fidelity = l2loss(R, data, weights);
                case 'mse'
                    loss_fidelity = mse(R, data);
            end
            
            % regularisation term
            if fitting.lambda > 0
                cost        = this.askadamObj.reg_TV(squeeze(parameters.(fitting.regmap)),squeeze(mask(1,:,:,:)),fitting.TVmode,fitting.voxelSize);
                loss_reg    = sum(abs(cost),"all")/fitting.Nsample *fitting.lambda;
            else
                loss_reg = 0;
            end
            
            % compute loss
            loss = loss_fidelity + loss_reg;

            for k = 1:numel(fitting.ub)
                parameters.(fitting.model_params{k}) = permute(parameters.(fitting.model_params{k}),[2 3 4 1]);
            end
            
            % Calculate gradients with respect to the learnable parameters.
            gradients = dlgradient(loss,parameters);
        
        end
        
        % compute weights for optimisation
        function w = compute_optimisation_weights(this,mask,dims,lossFunction,lmax)
        % 
        % Output
        % ------
        % w         : 1D signal masked wegiths
        %
            % lmax dependent weights
            l = 0:2:lmax;
            w = zeros(dims,'single');
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
            
            % mathcing dimension to the signal generated in modelGradients
            w       = permute(w,[4 1 2 3]);
            mask    = permute(mask,[4 1 2 3]);
            w = w(mask>0);
        end

        % compute rotationally invariant DWI signal if necessary
        function dwi = prepare_dwi_data(this,dwi,extradata,lmax)
            % full DWI data then compute rotaionally invariant signal
            if size(dwi,4)/(lmax/2+1) > numel(this.b) 
                % compute spherical mean signal
                fprintf('Computing rotationally invariant signal...')

                % if the inout little delta is one value then create a vector
                if numel(extradata.ldelta) == 1
                    extradata.ldelta = ones(size(extradata.bval)) * extradata.ldelta;
                end

                obj     = preparationDWI;
                [dwi]   = obj.get_Sl_all(dwi,extradata.bval,extradata.bvec,extradata.ldelta,extradata.BDELTA,lmax);

                fprintf('done.\n');

            elseif size(dwi,4) < numel(this.b)
                error('There are more b-shells in the class object than available in the input data. Please check your input data.');
            end
        end

        %% Prior estimation related functions
        % using maximum likelihood method to estimate starting points
        function pars0 = estimate_prior(this,dwi,mask, Nsample,lmax)
        % Estimation starting points for NEXI using likehood method

            start = tic;
            
            disp('Estimate starting points based on likelihood ...')

            % manage pool
            pool            = gcp('nocreate');
            isDeletepool    = false;
            if isempty(pool)
                Nworker = min(max(8,floor(maxNumCompThreads/4)),maxNumCompThreads);
                pool    = parpool('Processes',Nworker);
                isDeletepool = true;
            end

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
            if lmax == 0
                Nparam = 4;
            elseif lmax == 2
                Nparam = 5;
            end

            pars0_mask  = zeros(Nparam,length(ind));
            parfor kvol = 1:length(ind)
                pars0_mask(:,kvol) = this.likelihood(dwi(:,ind(kvol)), x_train, S_train,lmax);
            end
            pars0           = zeros(Nparam,size(dwi,2));
            pars0(:,ind)    = pars0_mask;

            % reshape estimation into image
            pars0           = permute(reshape(pars0,[size(pars0,1) dims(1:3)]),[2 3 4 1]);

            % Correction for CSF
            idx             = this.b < 4;
            D0              = this.b(idx)\-log(dwi(cat(1,idx,false(size(idx))),:));
            D0              = permute(reshape(D0,[size(D0,1) dims(1:3)]),[2 3 4 1]);
            D0(isnan(D0))   = 0;
            D0(isinf(D0))   = 0;
            D0(D0<0)        = 0;
            mask_CSF        = medfilt3(D0)>1;
            
            % ratio to modulate pars0 estimattion
            pars0_csf = [0.01,1,1,0.01,0.01];
            for k = 1:size(pars0,4)
                tmp                 = pars0(:,:,:,k);
                tmp(mask_CSF==1)    = tmp(mask_CSF==1).*pars0_csf(k);
                pars0(:,:,:,k)      = tmp;
            end

            ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
            fprintf('Starting points estimated. Elapsed time (hh:mm:ss): %s \n',string(ET));
            if isDeletepool
                delete(pool);
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

        %% NEXI signal related functions
        % compute the forward model
        function [s] = FWD(this, pars, fitting,mask)
        % Forward model to generate NEXI signal
            if nargin < 4
                fa   = pars.fa;
                Da   = pars.Da;
                De   = pars.De;
                ra   = pars.ra;
            else
                % mask out voxels to reduce memory
                fa   = pars.fa(mask(1,:,:,:)).';
                Da   = pars.Da(mask(1,:,:,:)).';
                De   = pars.De(mask(1,:,:,:)).';
                ra   = pars.ra(mask(1,:,:,:)).';
            end
                
            % Forward model
            % Sl0
            s = this.Sl0(fa, Da, De, ra);
            
            % Sl2
            if fitting.lmax == 2
                if nargin < 4
                    p2 = pars.p2;
                else
                    p2 = pars.p2(mask(1,:,:,:)).';
                end

                s = cat(1,s,this.Sl2(fa, Da, De, ra, p2));
            end

            % make sure s cannot be greater than 1
            s = min(s,1);
                
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
            re = ra.*fa./(1-fa);
            
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
            l2 = sqrt( (r1-r2+d1-d2).^2 + 4*r1.*r2 )/2; 
            lm = l1-l2;
            Pp = (fa.*d1 + (1-fa).*d2 - lm)./(l2*2);
            M  = Pp.*exp(-(l1+l2)) + (1-Pp).*exp(-lm); 
        end

        %% Utilities
        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default(fitting)
            % get basic fitting setting check
            fitting2 = askadam.check_set_default_basic(fitting);

            % get customised fitting setting check
            if ~-isfield(fitting,'regmap')
                fitting2.regmap = 'fa';
            end
            if ~isfield(fitting,'lmax')
                fitting2.lmax = 0;
            end

        end


    end

end