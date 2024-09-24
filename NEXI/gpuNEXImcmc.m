classdef gpuNEXImcmc < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 23 August 2024 (v0.1.0)
%

    properties
        % fa        : Intraneurite volume fraction
        % Da        : Intraneurite diffusivity (um2/ms)
        % De        : Extraneurite diffusivity (um2/ms)
        % ra        : exchange rate from intra- to extra-neurite compartment
        % p2        : dispersion index (if fitting.lax=2)
        % default model parameters and estimation boundary
        model_params    = { 'fa';   'Da';   'De';   'ra';'p2'; 'noise'};
        ub              = [    1;      3;      3;      1;   1;     0.1];
        lb              = [    0;  0.002;  0.001;  1/250;   0;    0.01];
        step            = [  0.05;  0.15;   0.15;  0.005;0.05;   0.005];
        startpoint      = [  0.2;      2;    0.5;   0.05; 0.2;    0.05];
    end

    properties (GetAccess = public, SetAccess = protected)
        b;
        Delta;  
        Nav;
    end
    
    methods

        % constructuor
        function this = gpuNEXImcmc(b, Delta, varargin)
        % NEXI Exchange rate estimation using NEXI model
        % obj = gpuNEXImcmc(b, Delta, Nav)
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
            
            this.b      = gpuArray( single(b(:)) );
            this.Delta  = gpuArray( single(Delta(:)) );
            if nargin > 2
                this.Nav = varargin{1} ;
            else
                this.Nav =  ones(size(b)) ;
            end
            this.Nav = gpuArray( single(this.Nav(:))) ;
        end
        
        % update properties according to lmax
        function this = updateProperty(this, fitting)

            if fitting.lmax == 0
                idx = find(ismember(this.model_params,'p2'));
                this.model_params(idx)    = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startpoint(idx)      = [];
                this.step(idx)            = [];
            end

        end

        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('================================================');
            disp('NEXI with Markov Chain Monte Carlo (MCMC) solver');
            disp('================================================');
            
            disp('----------------')
            disp('Data Information');
            disp('----------------')
            fprintf('b-shells (ms/um2)              : [%s] \n',num2str(this.b.',' %.2f'));
            fprintf('Diffusion time (ms)            : [%s] \n\n',num2str(this.Delta.',' %i'));

            fprintf('\n')

        end

        %% higher-level data fitting functions
        % Perform NEXI model parameter estimation based on MCMC across the whole dataset
        % This is a wrapper of the 'fit' function.
        % The main purpose of this function is to handle memory issue and ensure the input data is correct for 'fit'
        function  [out] = estimate(this, dwi, mask, extradata, fitting, pars0)
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
        % 
            
            % display basic info
            this.display_data_model_info;

            % get all fitting algorithm parameters 
            fitting = this.check_set_default(fitting);

            %%%%%%%%%%%%%%%% Step 1: Validate all input data %%%%%%%%%%%%%%%%
            % compute rotationally invariant signal if needed
            dwi = this.prepare_dwi_data(dwi,extradata,fitting.lmax);

            % mask sure no nan or inf
            [dwi,mask] = askadam.remove_img_naninf(dwi,mask);

            % if no pars input at all (not even empty) then use prior
            if nargin < 6; pars0 = []; end

            % convert datatype to single
            dwi     = single(dwi);
            mask    = mask >0;
            if ~isempty(pars0); for km = 1:numel(this.model_params); pars0.(this.model_params{km}) = single(pars0.(this.model_params{km})); end; end

            %%%%%%%%%%%%%%%% End Step 1 %%%%%%%%%%%%%%%%

            % MCMC main
            out      = this.fit(dwi, mask, fitting, pars0);
            out.mask = mask;

            % save the estimation results if the output filename is provided
            mcmc.save_mcmc_output(fitting.output_filename,out)

        end

        % Data fitting function, can be 2D (voxel-based) or 4D (image-based)
        function [out] = fit(this,dwi,mask,fitting,pars0)
        %
        % Input
        % -----------
        % dwi       : S0 normalised 4D dwi images, [x,y,slice,diffusion], 4th dimension corresponding to [Sl0_b1,Sl0_b2,Sl2_b1,Sl2_b2, etc.]; the order of bval must match the order in the constructor gpuNEXI
        % mask      : 3D signal mask, [x,y,slice]
        % fitting   : fitting algorithm parameters
        % pars0     : 4D parameter starting points of fitting, [x,y,slice,param], 4th dimension corresponding to fitting  parameters with order [fa,Da,De,ra,p2] (optional)
        % 
        % Output
        % -----------
        % out       : output structure
        %
        % Description: MCMC Image-based NEXI model fitting
        %
        % Kwok-Shing Chan @ MGH
        % kchan2@mgh.harvard.edu
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
                if ~isempty(pars0); for km = 1:numel(this.model_params); pars0.(this.model_params{km}) = single(pars0.(this.model_params{km})); end; end
            end

            % get all fitting algorithm parameters 
            fitting                 = this.check_set_default(fitting);
            % determine fitting parameters
            this                    = this.updateProperty(fitting);
            fitting.model_params    = this.model_params;
            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(fitting.model_params)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(fitting.model_params)); end
            fitting.xStepSize = this.step;

            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run MCMC and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w = this.compute_optimisation_weights(mask,fitting.lmax); % This is a customised funtion

            % You may add more dispay messages here
            disp('Model:')
            disp(['lmax                     = ' num2str(fitting.lmax)]);
            % additional message(s)
            if ischar(fitting.start); disp(['Starting points   : ', fitting.start ]); end; fprintf('\n');
            % set starting points
            if isempty(pars0);  pars0 = this.determine_x0(dwi,mask,fitting); end
            
            % 2.3 askAdam optimisation main
            mcmcObj = mcmc();
            out = mcmcObj.optimisation(dwi, mask, w, pars0, fitting, @this.FWD, fitting.lmax);

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            % clear GPU
            reset(gpool)
            
        end

        %% Data preparation

        % compute weights for optimisation
        function w = compute_optimisation_weights(this,mask,lmax)
        % 
        % Output
        % ------
        % w         : 1D signal masked wegiths
        %
            % lmax dependent weights
            l = 0:2:lmax;
            w = zeros([size(mask) numel(this.b)*numel(l)],'single');
            % w = zeros(dims,'single');
            for kl = 1:(lmax/2+1)
                for kb = 1:numel(this.b)
                    w(:,:,:,(kl-1)*numel(this.b)+kb) = this.Nav(kb) / (2*l(kl)+1);
                end
            end
            w = w ./ max(w(:));
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
                DWIutilityObj = DWIutility();
                [dwi]   = DWIutilityObj.get_Sl_all(dwi,extradata.bval,extradata.bvec,extradata.ldelta,extradata.BDELTA,lmax);

                fprintf('done.\n');

            elseif size(dwi,4) < numel(this.b)
                error('There are more b-shells in the class object than available in the input data. Please check your input data.');
            end
        end

        %%%%% Prior estimation related functions %%%%%
        % determine how the starting points will be set up
        function x0 = determine_x0(this,y,mask,fitting) 

            if ischar(fitting.start)
                switch lower(fitting.start)
                    case 'likelihood'
                        % using maximum likelihood method to estimate starting points
                        x0 = this.estimate_prior(y,mask,[],fitting.lmax);
    
                    case 'default'
                        % use fixed points
                        fprintf('Using default starting points for all voxels at [%s]: [%s]\n\n',mcmc.cell2str(this.model_params),replace(num2str(this.startpoint.',' %.2f'),' ',','));
                        fitting.start = this.startpoint;
                        x0 = mcmc.initialise_start(size(mask,1:3),fitting);
                    
                end
            else
                % user defined starting point
                x0 = fitting.start(:);
                fprintf('Using user-defined starting points for all voxels at [%s]: [%s]\n\n',mcmc.cell2str(this.model_params),replace(num2str(x0.',' %.2f'),' ',','));
                x0 = mcmc.initialise_start(dims,fitting);

            end
            fprintf('Estimation lower bound [%s]: [%s]\n',      mcmc.cell2str(this.model_params),replace(num2str(fitting.lb(:).',' %.2f'),' ',','));
            fprintf('Estimation upper bound [%s]: [%s]\n',      mcmc.cell2str(this.model_params),replace(num2str(fitting.ub(:).',' %.2f'),'  ',','));
            if strcmpi(fitting.algorithm,'mh')
                fprintf('MCMC step size [%s]: [%s]\n\n',            mcmc.cell2str(this.model_params),replace(num2str(this.step.',' %.2f'),'  ',','));
            end
        end

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
            pars           = zeros(Nparam,size(dwi,2));
            pars(:,ind)    = pars0_mask;

            % reshape estimation into image
            pars           = permute(reshape(pars,[size(pars,1) dims(1:3)]),[2 3 4 1]);

            % Correction for CSF
            bval_thres      = max(min(gather(this.b)),1.1);
            idx             = gather(this.b) <= bval_thres;
            D0              = real(this.b(idx)\-log(dwi(cat(1,idx,false(size(idx))),:)));
            D0              = permute(reshape(D0,[size(D0,1) dims(1:3)]),[2 3 4 1]);
            D0(isnan(D0))   = 0;
            D0(isinf(D0))   = 0;
            D0(D0<0)        = 0;
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
                pars0.(this.model_params{km}) = pars(:,:,:,km); 
            end

            % noise
            pars0.(this.model_params{end}) = single(ones(size(mask)) * this.startpoint(end));

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
        function [s] = FWD(this, pars, lmax)
            % make sure the first dimension is 1
            if size(pars.fa,1) ~= 1
                fa   = shiftdim(pars.fa,-1);
                Da   = shiftdim(pars.Da,-1);
                De   = shiftdim(pars.De,-1);
                ra   = shiftdim(pars.ra,-1);
            else
                % mask out voxels to reduce memory
                fa   = pars.fa;
                Da   = pars.Da;
                De   = pars.De;
                ra   = pars.ra;
            end
                
            % Forward model
            if lmax == 2
                if size(pars.fa,1) ~= 1
                    p2 = shiftdim(pars.p2,-1);
                else
                    p2 = pars.p2;
                end
            else
                p2 = [];
            end
            s = this.Slmax2(fa, Da, De, ra, p2);

            % make sure s cannot be greater than 1
            s = min(s,1);
                
        end
        
        function S = Slmax2(this, fa, Da, De, ra, p2)

            bval    = this.b;
            DELTA   = this.Delta;
            
            Da = bval.*Da;
            De = bval.*De;
            ra = DELTA.*ra;
            re = ra.*fa./(1-fa);

            % Trapezoidal's rule replacement
            Nx  = 14;    % NRMSE<0.05% for Nx=14
            x   = zeros([ones(1,ndims(fa)), Nx],'like',bval); x(:) = linspace(0,1,Nx); dx = x(2) - x(1);

            M = arrayfun(@NEXI_M,x,fa,Da,De,ra,re);

            % Sl0
            % bypass Matlab's trapz for speed
            if ndims(M) == 3
                S = sum((M(:,:,2:end) + M(:,:,1:end-1)) * (dx) / 2, ndims(x));
            elseif ndims(M) == 4
                S = sum((M(:,:,:,2:end) + M(:,:,:,1:end-1)) * (dx) / 2, ndims(x));
            end

            if ~isempty(p2)
                % M = M.*(3*x.^2-1)/2; 
                M = arrayfun(@NEXI_MSl2,M,x);
                % bypass Matlab's trapz for speed
                if ndims(M) == 3
                    Sl2 = sum((M(:,:,2:end) + M(:,:,1:end-1)) * (dx) / 2, ndims(x));
                elseif ndims(M) == 4
                    Sl2 = sum((M(:,:,:,2:end) + M(:,:,:,1:end-1)) * (dx) / 2, ndims(x));
                end
                Sl2 = p2.*abs(Sl2);

                S = cat(1,S,Sl2);
            end

        end

        % 0th order rotational invariant
        function S = Sl0(this, fa, Da, De, ra)

            if isgpuarray(fa)
                bval    = gpuArray(single(this.b));
                DELTA   = gpuArray(single(this.Delta));
            else
                bval  = this.b;
                DELTA = this.Delta;
            end

            Da = bval.*Da;
            De = bval.*De;
            ra = DELTA.*ra;
            re = ra.*fa./(1-fa);
            
            % Trapezoidal's rule replacement
            Nx  = 14;    % NRMSE<0.05% for Nx=14
            x   = zeros([ones(1,ndims(fa)), Nx],'like',bval); x(:) = linspace(0,1,Nx); dx = x(2) - x(1);

            % S   = trapz(x(:),this.M(x, fa, Da, De, ra, re),ndims(x));
            % S   = trapz(x(:), arrayfun(@NEXI_M,x,fa,Da,De,ra,re),ndims(x));
            M = arrayfun(@NEXI_M,x,fa,Da,De,ra,re);
            % bypass Matlab's trapz for speed
            S = sum((M(:,:,2:end) + M(:,:,1:end-1)) * (dx) / 2, ndims(x));

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
            x   = zeros([ones(1,ndims(fa)), Nx],'like',bval); x(:) = linspace(0,1,Nx); dx = x(2) - x(1);
            % S   = trapz(x(:),this.M(x, fa, Da, De, ra, re).*(3*x.^2-1)/2,ndims(x));
            % S   = trapz(x(:), arrayfun(@NEXI_M,x,fa,Da,De,ra,re).*(3*x.^2-1)/2,ndims(x));
            M = arrayfun(@NEXI_M,x,fa,Da,De,ra,re).*(3*x.^2-1)/2; 
            % bypass Matlab's trapz for speed
            S = sum((M(:,:,2:end) + M(:,:,1:end-1)) * (dx) / 2, ndims(x));

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
            fitting2 = mcmc.check_set_default_basic(fitting);

            % get customised fitting setting check
            if ~isfield(fitting,'lmax')
                fitting2.lmax = 0;
            end

        end

    end

end