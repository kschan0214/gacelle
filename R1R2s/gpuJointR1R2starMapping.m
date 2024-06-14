classdef gpuJointR1R2starMapping
% This is the method to perform joint R1-R2* mapping using variable flip angle (vFA), multiecho GRE (mGRE) data
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 5 April 2024
% Date modified: 

    properties
        % default model parameters and estimation boundary
        model_params    = {'M0','R1','R2star'};
        ub              = [  20,  10, 200];
        lb              = [ eps, 0.1, 0.1];
    end

    properties (GetAccess = public, SetAccess = protected)
        te;
        tr;
        fa;
        askadamObj;
    end
    
    methods

        % constructuor
        function this = gpuJointR1R2starMapping(te,tr,fa)
            this.te = single(te(:));
            this.tr = single(tr(:));
            this.fa = single(fa(:));

            this.askadamObj = askadam();

        end
        
        %% higher-level data fitting functions
        % Wrapper function of fit to handle image data; automatically segment data and fitting in case the data cannot fit in the GPU in one go
        function  [out, M0, R1, R2star] = estimate(this, data, mask, extradata, fitting)
        % Perform NEXI model parameter estimation based on askAdam
        % Input data are expected in multi-dimensional image
        % 
        % Input
        % -----------
        % dwi       : 4D DWI, [x,y,z,dwi]
        % mask      : 3D signal mask, [x,y,z]
        % extradata : Optional additional data
        %   .b1map  : 3D B1 map, [x,y,z]
        % fitting   : fitting algorithm parameters (see fit function)
        % 
        % Output
        % -----------
        % out       : output structure contains all estimation results
        % M0        : Proton desity weighted signal
        % R1        : R1 map
        % R2star    : R2* map
        % 
            
            disp('========================================================');
            disp('Joint R1-R2* mapping with vFA-mGRE data - askAdam solver');
            disp('========================================================');

            disp('----------------')
            disp('Data Information');
            disp('----------------')
            fprintf('Echo time (TE) (ms)             : [%s] \n',num2str(this.te.' * 1e3,' %.2f'));
            fprintf('Repetition time (TR) (ms)       : [%s] \n',num2str(this.tr.' * 1e3,' %.2f'));
            fprintf('Flip angle (degree)             : [%s] \n\n',num2str(this.fa.',' %i'));

            % get all fitting algorithm parameters 
            fitting             = this.check_set_default(fitting);
            [mask,extradata]    = this.validate_input(data,mask,extradata);

            % mask sure no nan or inf
            [data,mask] = this.askadamObj.remove_img_naninf(data,mask);

            % get matrix size
            dims = size(data);

            if fitting.isNormData; [scaleFactor, data] = MWIutility.image_normalisation_ND(data,mask);
            else;                   scaleFactor = 1; end

            % check prior input
            if fitting.isPrior
                pars0 = this.estimate_prior(data,mask, extradata.b1map);
            end

            % convert datatype to single
            data    = single(data);
            mask    = mask >0;
            if ~isempty(pars0); for km = 1:numel(this.model_params); pars0.(this.model_params{km}) = single(pars0.(this.model_params{km})); end; end
            
            % determine if we need to divide the data to fit in GPU
            g = gpuDevice; reset(g);
            memoryFixPerVoxel       = 0.0013/3;   % get this number based on mdl fit
            memoryDynamicPerVoxel   = 0.05/3;     % get this number based on mdl fit
            [NSegment,maxSlice]     = this.askadamObj.find_optimal_divide(mask,memoryFixPerVoxel,memoryDynamicPerVoxel);

            % parameter estimation
            out     = [];
            M0      = zeros(dims(1:3),'single');
            R1      = zeros(dims(1:3),'single');
            R2star  = zeros(dims(1:3),'single');
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
                data_tmp    = data(:,:,slice,:,:);
                mask_tmp    = mask(:,:,slice);
                if ~isempty(pars0); for km = 1:numel(this.model_params); pars0_tmp.(this.model_params{km}) = pars0.(this.model_params{km})(:,:,slice); end
                else;               pars0_tmp = [];                 end
                fieldname = fieldnames(extradata);
                for kfield = 1:numel(fieldname)
                    extradata_tmp.(fieldname{kfield}) = extradata.(fieldname{kfield}) (:,:,slice);
                end

                % run fitting
                [out_tmp, M0(:,:,slice), R1(:,:,slice), R2star(:,:,slice)] = this.fit(data_tmp,mask_tmp,extradata_tmp,fitting,pars0_tmp);

                % restore 'out' structure from segment
                out = this.askadamObj.restore_segment_structure(out,out_tmp,slice,ks);

            end
            out.mask        = mask;
            out.min.M0      = out.min.M0 * scaleFactor; % undo scaling
            out.final.M0    = out.final.M0 * scaleFactor; % undo scaling
            M0              = M0 * scaleFactor;      % undo scaling

            % save the estimation results if the output filename is provided
            this.askadamObj.save_askadam_output(fitting.output_filename,out)

        end

        % Data fitting function, can be 3D (voxel-based) or 5D (image-based)
        function [out, M0, R1, R2star] = fit(this, data, mask, extradata, fitting,pars0)
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
            dims = size(data);

            %%%%%%%%%%%%%%%%%%%% 1. Validate and parse input %%%%%%%%%%%%%%%%%%%%
            if nargin < 3 || isempty(mask)
                % if no mask input then fit everthing
                mask = ones(dims);
            else
                % if mask is 3D
                if ndims(mask) < 4
                    mask = repmat(mask,[1 1 1 dims(4) dims(5)]);
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
                for km = 1:numel(this.model_params); pars0.(this.model_params{km}) = single(pars0.(this.model_params{km})); end
            end

            % get all fitting algorithm parameters 
            fitting         = this.check_set_default(fitting);
            fitting.Nsample = numel(mask(mask ~= 0)) / prod(dims(4:end)); 

            % determine how many model parameters
            fitting.model_params = this.model_params;

            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub; end
            if isempty( fitting.lb); fitting.lb = this.lb; end
            % more customisation on the upper bound here
            if ~isempty(pars0); tmp = pars0.M0;
            else;               tmp = max(max(data,[],4),5,[],5)*20;  end % not ideal but it is what it is
            rho_max         = prctile(tmp(mask(:,:,:,1,1)),99);
            fitting.ub(1)   = rho_max; % input dependent value

            % put data input gpuArray
            mask        = gpuArray(logical(mask));  
            data        = gpuArray(single(data));
            true_famp   = gpuArray(single( extradata.b1map .* permute(deg2rad(this.fa),[2 3 4 5 1]) ));
            
            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w = this.compute_optimisation_weights(data,mask,fitting); % This is a tailored funtion
            w = dlarray(gpuArray(w).','CB');

            % 2.2 display optimisation algorithm parameters
            this.display_algorithm_info(fitting)

            % 2.3 askAdam optimisation main
            out = this.askadamObj.optimisation(pars0, @this.modelGradients, data, mask, w, fitting, true_famp);

            % 2.4 organise output
            M0      = out.final.M0;
            R1      = out.final.R1;
            R2star  = out.final.R2star;

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            disp('The process is completed.')
            
            % clear GPU
            reset(g)
            
        end

        % compute the gradient and loss of forward modelling
        function [gradients,loss,loss_fidelity,loss_reg] = modelGradients(this, parameters, data, mask, weights, fitting, true_famp)
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
        % true_famp     : true flip angle map, in radian
        % 
            % rescale network parameter to true values
            parameters = this.askadamObj.rescale_parameters(parameters,fitting.lb,fitting.ub,fitting.model_params);
            
            % Forward model
            % R           = this.FWD(parameters,fitting);
            R           = this.FWD(parameters,true_famp,mask(:,:,:,1,1));
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
                case 'huber'
                    loss_fidelity = huber(R, data, weights);
            end
            
            % regularisation term
            if fitting.lambda > 0
                cost        = this.askadamObj.reg_TV(squeeze(parameters.(fitting.regmap)),squeeze(mask(:,:,:,1,1)),fitting.TVmode,fitting.voxelSize);
                loss_reg    = sum(abs(cost),"all")/fitting.Nsample *fitting.lambda;
            else
                loss_reg = 0;
            end
            
            % compute loss
            loss = loss_fidelity + loss_reg;
            
            % Calculate gradients with respect to the learnable parameters.
            gradients = dlgradient(loss,parameters);
        
        end
        
        %% Prior estimation related functions
        % closed-form solution to estimate better starting points
        function pars0 = estimate_prior(this,data,mask, b1map)

            start = tic;
            
            disp('Estimate starting points using closed-form solutions...')

            % R2* closed-form solution
            r2s0            = this.R2star_trapezoidal(mean(abs(data),5),this.te);
            mask_valida_r2s = and(~isnan(r2s0),~isinf(r2s0));
            r2s0(mask_valida_r2s == 0) = 0;

            % DESPOT1 linear inversion
            despot1Obj  = despot1(this.tr,this.fa);
            [t10, m00]  = despot1Obj.estimate(permute(abs(data(:,:,:,1,:)),[1 2 3 5 4]),mask,b1map);
            R10 = 1./t10;

            % smooth out outliers
            m00    = medfilt3(m00);
            R10    = medfilt3(R10);
            r2s0   = medfilt3(r2s0);

            % always follow the order specified in the beginning of the file
            pars0.(this.model_params{1}) = m00; 
            pars0.(this.model_params{2}) = R10;
            pars0.(this.model_params{3}) = r2s0;
            % pars0 = cat(4, m00, R10, r2s0);

            ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
            fprintf('Starting points estimated. Elapsed time (hh:mm:ss): %s \n',string(ET));
  
        end

        %% Signal related functions
        % compute the forward model
        function [s] = FWD(this, pars, true_famp, mask)
            % TE in 4th dimension
            TE  = permute(this.te,[2 3 4 1 5]);
            
            if nargin < 4
                M0      = pars.M0;
                R1      = pars.R1;
                R2star  = pars.R2star;
            else
                % mask out voxels to reduce memory
                dims = size(true_famp);
                M0          = pars.M0(mask);
                R1          = pars.R1(mask);
                R2star      = pars.R2star(mask);
                true_famp   = reshape(true_famp,[prod(dims(1:3)),1,1, dims(4) , dims(5)]);
                true_famp   = true_famp(mask,:,:,:,:);
            end
            
            s = this.model_jointR1R2s(M0, R2star, R1, TE,this.tr,true_famp);
            % vectorise to match maksed measurement data
            s = s(:);
                
        end
        
        %% utility
        function [mask,extradata] = validate_input(this,data,mask,extradata)
            %%%%%%%%%% 2. check data integrity %%%%%%%%%%
            disp('-----------------------');
            disp('Checking data integrity');
            disp('-----------------------');
            % check if the number of echo times matches with the data
            if numel(this.te) ~= size(data,4)
                error('The size of TE does not match with the 4th dimension of the image.');
            end
            if numel(this.fa) ~= size(data,5)
                error('The size of flip angle does not match with the 5th dimension of the image.');
            end
            % check signal mask
            if ~isempty(mask)
                disp('Mask input                : True');
                if min(size(data(:,:,:,1,1)) == size(mask)) == 0
                    error('The dimension of the mask does not match the inpt image.');
                end
            else
                disp('Mask input                : False');
                disp('Default masking method is used.');

                % if no mask input then use default method to generate a signal mask
                mask = max(max(abs(data),[],4),[],5)./max(abs(data(:))) > 0.05;
            end
            if ~isempty(extradata)
                disp('B1map input               : True');

                if ~isfield(extradata,'b1map')
                    error('Cannot find B1 map in extradata structure. Please specify B1 map to ''extradata.b1map = b1map'' or set extradata to empty: ''extradata = []''');
                end

                if min(size(data(:,:,:,1,1)) == size(extradata.b1map)) == 0
                    error('The dimension of the B1 map does not match the inpt image.');
                end
            else
                disp('B1map input              : False');
                extradata.b1map = ones(size(mask));
            end

            disp('Input data is valid.')
        end

         % display fitting algorithm information
        function display_algorithm_info(this,fitting)
        
            this.askadamObj.display_basic_fitting_parameters(fitting);

            % You may add more dispay messages here
            if fitting.isNormData;  disp('GRE data is normalised before fitting');
            else;                   disp('GRE data is not normalised before fitting'); end

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

    end

    methods(Static)
        %% Signal related
        function signal = model_jointR1R2s(m0,r2s,R1,te,TR,fa)
        % m0    : proton density weighted signal
        % r2s   : R2*, in s^-1 or ms^-1
        % R1    : R1, in s^-1 or ms^-1
        % te    : echo time, in s or ms
        % TR    : repetition time, in s or ms
        % fa    : true flip angles, in radian

            signal = m0 .* sin(fa) .* (1 - exp(-TR.*R1)) ./ (1 - cos(fa) .* exp(-TR.*R1)) .* ...
                            exp(-te .* r2s);
        
        
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
            % get basic fitting setting check
            fitting2 = askadam.check_set_default_basic(fitting);

            % get customised fitting setting check
            if ~-isfield(fitting,'isNormData')
                fitting2.isNormData = true;
            end
            if ~isfield(fitting,'isWeighted')
                fitting2.isWeighted = true;
            end
            if ~isfield(fitting,'weightPower')
                fitting2.weightPower = 2;
            end

        end

        % compute weights for optimisation
        function w = compute_optimisation_weights(data,mask,fitting)
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
                        % weights using the 1st echo intensity of each flip angle
                        w       = bsxfun(@rdivide,abs(data).^p,abs(data(:,:,:,1,:)).^p);
                        w(w>1)  = 1;
                end
            else
                % compute the cost without weights
                w = ones(size(data));
            end
            w = w(mask>0);
        end

    end

end