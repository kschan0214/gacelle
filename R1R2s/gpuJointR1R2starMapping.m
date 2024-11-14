classdef gpuJointR1R2starMapping < handle
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
        modelParams     = {'M0';'R1';'R2star'};
        ub              = [  2;   10;  200];
        lb              = [  0;  0.1;  0.1];
        startPoint      = [   1;   1;  30];
    end

    properties (GetAccess = public, SetAccess = protected)
        te;
        tr;
        fa;
    end
    
    methods

        % constructuor
        function this = gpuJointR1R2starMapping(te,tr,fa)

            this.te = single(te(:));
            this.tr = single(tr(:));
            this.fa = single(fa(:));

        end
        
        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('========================================================');
            disp('Joint R1-R2* mapping with vFA-mGRE data - askAdam solver');
            disp('========================================================');

            
            disp('----------------')
            disp('Data Information');
            disp('----------------')
            fprintf('Echo time (TE) (ms)             : [%s] \n',num2str(this.te.' * 1e3,' %.2f'));
            fprintf('Repetition time (TR) (ms)       : [%s] \n',num2str(this.tr.' * 1e3,' %.2f'));
            fprintf('Flip angle (degree)             : [%s] \n\n',num2str(this.fa.',' %i'));
            disp('----------------')

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

            % get matrix size
            dims = size(data,1:3);

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
            g = gpuDevice; reset(g);
            memoryFixPerVoxel       = 0.0013/3;   % get this number based on mdl fit
            memoryDynamicPerVoxel   = 0.05/3;     % get this number based on mdl fit
            [NSegment,maxSlice]     = utils.find_optimal_divide(mask,memoryFixPerVoxel,memoryDynamicPerVoxel);

            % parameter estimation
            out     = [];
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
                fields      = fieldnames(extraData); for kfield = 1:numel(fields); extraData_tmp.(fields{kfield}) = extraData.(fields{kfield})(:,:,slice,:,:,:,:); end

                % run fitting
                [out_tmp] = this.fit(data_tmp,mask_tmp,fitting,extraData_tmp);

                % restore 'out' structure from segment
                out = utils.restore_segment_structure(out,out_tmp,slice,ks);

            end
            out.mask        = mask;
            out.min.M0      = out.min.M0 * scaleFactor; % undo scaling
            out.final.M0    = out.final.M0 * scaleFactor; % undo scaling

            % save the estimation results if the output filename is provided
            askadam.save_askadam_output(fitting.outputFilename,out)

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
            fitting.modelParams     = this.modelParams;
            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(this.modelParams)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(this.modelParams)); end

            % set initial tarting points
            pars0 = this.determine_x0(data,mask,extraData,fitting) ;

            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w = this.compute_optimisation_weights(data,fitting); % This is a tailored funtion

            % 2.2 display optimisation algorithm parameters
            this.display_algorithm_info(fitting)

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            % 3. askAdam optimisation main
            askadamObj  = askadam();
            % 3.1. initial global optimisation
            extraData   = utils.gpu_reshape_ND2GD_struct(extraData,mask);
            out         = askadamObj.optimisation(data, mask, w, pars0, fitting, @this.FWD, extraData);

            disp('The process is completed.')
            
            % clear GPU
            reset(gpool)
            
        end
        
        %% Prior estimation related functions

        % determine how the starting points will be set up
        function x0 = determine_x0(this,y,mask,extraData,fitting) 

            disp('---------------');
            disp('Starting points');
            disp('---------------');

            dims = size(mask,1:3);

            if ischar(fitting.start)
                switch lower(fitting.start)
                    case 'prior'
                        % using maximum likelihood method to estimate starting points
                        x0 = this.estimate_prior(y,mask,extraData);
    
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

        % closed-form solution to estimate better starting points
        function pars0 = estimate_prior(this,data, mask, extraData)

            dims = size(data,1:3);

            for k = 1:numel(this.modelParams)
                pars0.(this.modelParams{k}) = single(this.startPoint(k)*ones(dims));
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
            % % smooth out outliers
            % m00    = medfilt3(m00);
            % R10    = medfilt3(R10);
            % r2s0   = medfilt3(r2s0);

            % always follow the order specified in the beginning of the file
            pars0.(this.modelParams{1}) = single(m00); 
            pars0.(this.modelParams{2}) = single(R10);
            pars0.(this.modelParams{3}) = single(R2s0);
            % pars0 = cat(4, m00, R10, r2s0);

            ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
            fprintf('Starting points estimated. Elapsed time (hh:mm:ss): %s \n',string(ET));
  
        end

        %% Signal related functions
        % compute the forward model
        function [s] = FWD(this, pars, extraData)
            
            TE  = gpuArray(dlarray( permute(this.te,[2 3 4 1 5]))); % TE in 4th dimension
            FA  = gpuArray(dlarray( permute(deg2rad(this.fa),[2 3 4 5 1]))); % FA in 5th dimension

            if ~isfield(extraData,'trueFlipAngle')
                trueFlipAngle = extraData.b1 .* FA;
            else
                trueFlipAngle = extraData.trueFlipAngle;
            end
            
            M0      = pars.M0;
            R1      = pars.R1;
            R2star  = pars.R2star;
            % R2star(R1>R2star) = R1(R1>R2star);
            
            s = this.model_jointR1R2s(M0, R2star, R1, TE,this.tr,trueFlipAngle);
            % vectorise to match maksed measurement data
            s = utils.reshape_ND2GD(s,[]);
                
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
            fitting2 = askadam.check_set_default_basic(fitting);

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