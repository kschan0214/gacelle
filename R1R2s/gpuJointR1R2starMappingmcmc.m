classdef gpuJointR1R2starMappingmcmc < handle
% This is the method to perform joint R1-R2* mapping using variable flip angle (vFA), multiecho GRE (mGRE) data
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 5 April 2024
% Date modified: 25 September 2024

    properties
        % default model parameters and estimation boundary
        % M0    : Proton density weighted signal
        % R1    : (=1/T1) in s^-1
        % R2star: R2* in s^-1   
        model_params    = {'M0';'R1';'R2star';'noise'};
        ub              = [   2;  10;     200;    0.1];
        lb              = [   0; 0.1;     0.1;  0.001];
        startpoint      = [   1;   1;      30;   0.05];
        step            = [0.01;0.01;       1;  0.005];
    end

    properties (GetAccess = public, SetAccess = protected)
        te;
        tr;
        fa;
    end
    
    methods

        % constructuor
        function this = gpuJointR1R2starMappingmcmc(te,tr,fa)

            this.te = single(te(:));
            this.tr = single(tr(:));
            this.fa = single(fa(:));

        end
        
        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('===================================================================================');
            disp('Joint R1-R2* mapping with vFA-mGRE data with Markov Chain Monte Carlo (MCMC) solver');
            disp('===================================================================================');

            
            disp('----------------')
            disp('Data Information');
            disp('----------------')
            fprintf('Echo time (TE) (ms)             : [%s] \n',num2str(this.te.' * 1e3,' %.2f'));
            fprintf('Repetition time (TR) (ms)       : [%s] \n',num2str(this.tr.' * 1e3,' %.2f'));
            fprintf('Flip angle (degree)             : [%s] \n\n',num2str(this.fa.',' %i'));
            
            fprintf('\n')

        end

        %% higher-level data fitting functions
        % Wrapper function of fit to handle image data; automatically segment data and fitting in case the data cannot fit in the GPU in one go
        function  [out] = estimate(this, data, mask, extraData, fitting)
        % Perform NEXI model parameter estimation based on askAdam
        % Input data are expected in multi-dimensional image
        % 
        % Input
        % -----------
        % dwi       : 4D DWI, [x,y,z,dwi]
        % mask      : 3D signal mask, [x,y,z]
        % extradata : Optional additional data
        %   .b1  : 3D B1 map, [x,y,z]
        % fitting   : fitting algorithm parameters (see fit function)
        % 
        % Output
        % -----------
        % out       : output structure contains all estimation results
        % M0        : Proton desity weighted signal
        % R1        : R1 map
        % R2star    : R2* map
        % 

            % display basic info
            this.display_data_model_info;
            
            % get all fitting algorithm parameters 
            fitting             = this.check_set_default(fitting);

            % % get matrix size
            % dims = size(data,1:3);

            % make sure input data are valid
            [mask,extraData]    = this.validate_input(data,mask,extraData);

            % normalised data if needed
            [data, scaleFactor] = this.prepare_data(data,mask, extraData.b1);

            % mask sure no nan or inf
            [data,mask] = utils.remove_img_naninf(data,mask);

            % convert datatype to single
            data    = single(data);
            mask    = mask > 0;

            % determine if we need to divide the data to fit in GPU
            % MCMC main
            out      = this.fit(data, mask, fitting, extraData);
            out.mask = mask;
            for k = 1:numel(fitting.metric)
                out.(fitting.metric{k}).M0 = out.(fitting.metric{k}).M0 *scaleFactor;
            end
            out.posterior.M0 = out.posterior.M0 *scaleFactor;

            % g = gpuDevice; reset(g);
            % memoryFixPerVoxel       = 0.0013/3;   % get this number based on mdl fit
            % memoryDynamicPerVoxel   = 0.05/3;     % get this number based on mdl fit
            % [NSegment,maxSlice]     = askadam.find_optimal_divide(mask,memoryFixPerVoxel,memoryDynamicPerVoxel);

            % parameter estimation
            % out     = [];
            % for ks = 1:NSegment
            % 
            %     fprintf('Running #Segment = %d/%d \n',ks,NSegment);
            %     disp   ('------------------------')
            % 
            %     % determine slice# given a segment
            %     if ks ~= NSegment
            %         slice = 1+(ks-1)*maxSlice : ks*maxSlice;
            %     else
            %         slice = 1+(ks-1)*maxSlice : dims(3);
            %     end
            % 
            %     % divide the data
            %     data_tmp    = data(:,:,slice,:,:);
            %     mask_tmp    = mask(:,:,slice);
            %     fields      = fieldnames(extraData); for kfield = 1:numel(fields); extraData_tmp.(fields{kfield}) = extraData.(fields{kfield})(:,:,slice,:,:); end
            % 
            %     % run fitting
            %     [out_tmp] = this.fit(data_tmp,mask_tmp,fitting,extraData_tmp);
            % 
            %     % restore 'out' structure from segment
            %     out = askadam.restore_segment_structure(out,out_tmp,slice,ks);
            % 
            % end
            % out.mask        = mask;
            % out.min.M0      = out.min.M0 * scaleFactor; % undo scaling
            % out.final.M0    = out.final.M0 * scaleFactor; % undo scaling

            % save the estimation results if the output filename is provided
            mcmc.save_mcmc_output(fitting.output_filename,out)

        end

        % Data fitting function, can be 3D (voxel-based) or 5D (image-based)
        function [out] = fit(this, data, mask, fitting, extraData)
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
        %       .loss       : final loss metric
        %   .min        : results with the minimum loss metric across all iterations
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
            dims = size(data,1:3);

            %%%%%%%%%%%%%%%%%%%% 1. Validate and parse input %%%%%%%%%%%%%%%%%%%%
            if nargin < 3 || isempty(mask); mask = ones(dims,'logical'); end % if no mask input then fit everthing
            if nargin < 4; fitting = struct(); end

            % get all fitting algorithm parameters 
            fitting                 = this.check_set_default(fitting);
            % determine fitting parameters
            fitting.model_params    = this.model_params;
            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(this.model_params)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(this.model_params)); end

            % set initial tarting points
            pars0 = this.estimate_prior(data,mask,extraData);

            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w = this.compute_optimisation_weights(data,fitting); % This is a tailored funtion

            % 2.2 display optimisation algorithm parameters
            this.display_algorithm_info(fitting)

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            % 3. askAdam optimisation main
            % 3.1. initial global optimisation
            % mask out data to reduce memory load
            fieldname   = fieldnames(extraData); for km = 1:numel(fieldname); extraData.(fieldname{km}) = gpuArray(single( utils.vectorise_NDto2D(extraData.(fieldname{km}),mask) ).'); end
            mcmcObj     = mcmc();
            out         = mcmcObj.optimisation(data, mask, w, pars0, fitting, @this.FWD, fitting, extraData);

            % disp('The process is completed.')
            
            % clear GPU
            reset(gpool)
            
        end
        
        %% Prior estimation related functions
        % closed-form solution to estimate better starting points
        function pars0 = estimate_prior(this,data, mask, extraData)

            dims = size(data,1:3);

            for k = 1:numel(this.model_params)
                pars0.(this.model_params{k}) = single(this.startpoint(k)*ones(dims));
            end

            disp('Estimate starting points using closed-form solutions...')
            
            start = tic;
            % R2* closed-form solution
            R2s0            = this.R2star_trapezoidal(mean(abs(data),5),this.te);
            mask_valid = and(~isnan(R2s0),~isinf(R2s0));
            R2s0(mask_valid == 0) = this.lb(3); R2s0(R2s0<this.lb(3)) = this.lb(3); R2s0(R2s0>this.ub(3)) = this.ub(3);

            % DESPOT1 linear inversion
            despot1Obj  = despot1(this.tr,this.fa);
            [t10, m00]  = despot1Obj.estimate(permute(abs(data(:,:,:,1,:)),[1 2 3 5 4]),mask,extraData.b1);
            R10 = 1./t10;
            mask_valid = and(~isnan(m00),~isinf(m00));
            m00(mask_valid == 0) = this.lb(1); m00(m00<this.lb(1)) = this.lb(1); m00(m00>this.ub(1)) = this.ub(1);
            mask_valid = and(~isnan(R10),~isinf(R10));
            R10(mask_valid == 0) = this.lb(2); R10(R10<this.lb(2)) = this.lb(2); R10(R10>this.ub(2)) = this.ub(2);
            
            R2s0(R10>R2s0) = R10(R10>R2s0)/2;

            % always follow the order specified in the beginning of the file
            pars0.(this.model_params{1}) = single(m00); 
            pars0.(this.model_params{2}) = single(R10);
            pars0.(this.model_params{3}) = single(R2s0);

            ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
            fprintf('Starting points estimated. Elapsed time (hh:mm:ss): %s \n',string(ET));
  
        end

        %% Signal related functions
        % compute the forward model
        function [s] = FWD(this, pars, fitting, extraData)
            
            TE  = gpuArray( permute(this.te,[2 3 4 1 5]));             % TE in 4th dimension
            FA  = gpuArray( permute(deg2rad(this.fa),[2 3 4 5 1]));    % FA in 5th dimension

            if ~isfield(extraData,'trueFlipAngle')
                trueFlipAngle = extraData.b1 .* FA;
            else
                trueFlipAngle = extraData.trueFlipAngle;
            end
            
            M0      = pars.M0;
            R1      = pars.R1;
            R2star  = pars.R2star;
            
            % s = this.model_jointR1R2s(M0, R2star, R1, TE,this.tr,trueFlipAngle);
            s = arrayfun(@model_jointR1R2s_singlecompartment,M0, R2star, R1, TE,this.tr,trueFlipAngle);
            % vectorise to match maksed measurement data
            s = utils.vectorise_NDto2D(s,ones(size(M0),'logical')).';
            % reshape s for GW
            if ~isempty(fitting)
                if strcmpi(fitting.algorithm,'gw')
                    s = reshape(s, [size(s,1) size(s,2)/fitting.Nwalker fitting.Nwalker]);
                end
            end
                
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

                if ~isfield(extradata,'b1')
                    error('Cannot find B1 map in extradata structure. Please specify B1 map to ''extradata.b1 = b1map'' or set extradata to empty: ''extradata = []''');
                end

                if min(size(data(:,:,:,1,1)) == size(extradata.b1)) == 0
                    error('The dimension of the B1 map does not match the inpt image.');
                end
            else
                disp('B1map input              : False');
                extradata.b1 = ones(size(mask));
            end

            disp('Input data is valid.')
        end

        % normalise input data based on masked signal intensity at 98%
        function [img, scaleFactor] = prepare_data(this,img, mask, b1)

            despot1_obj          = despot1(this.tr,this.fa);
            [~, m0, mask_fitted] = despot1_obj.estimate(permute(abs(img(:,:,:,1,:)),[1 2 3 5 4]), mask, b1);

            scaleFactor = prctile( m0(mask_fitted>0), 98);

            img = img ./ scaleFactor;

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
            fitting2 = mcmc.check_set_default_basic(fitting);

            % get customised fitting setting check
            if ~isfield(fitting,'weightMethod');        fitting2.weightMethod   = '1stecho';    end
            if ~isfield(fitting,'isWeighted');          fitting2.isWeighted = false;            end
            if ~isfield(fitting,'weightPower');         fitting2.weightPower = 2;               end

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
                        % weights using the 1st echo intensity of each flip angle
                        w       = bsxfun(@rdivide,abs(data).^p,abs(data(:,:,:,1,:)).^p);
                end
            else
                % compute the cost without weights
                w = ones(size(data),'single');
            end
            w(w>1) = 1; w(w<0) = 0;
        end

    end

end