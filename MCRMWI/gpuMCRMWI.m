classdef gpuMCRMWI
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 
% Date modified: 

    properties (Constant)
            gyro = 42.57747892;
    end

    % here define the fitting parameters and their conditions
    properties
        % default model parameters and estimation boundary
        % S0        : T1w signal [a.u.] 
        % MWF       : myelin water fraction [0,1]
        % IWF       : intracellular volume ratio (=Vic or ICVF in DWI) [0,1]
        % R2sMW     : R2* MW [1/s]
        % R2sIW     : R2* IW [1/s] 
        % R2sEW     : R2* EW [1/s] 
        % freqMW    : frequency MW [ppm]
        % freqIW    : frequency IW [ppm]
        % dfreqBKG  : background frequency in addition to the one provided [ppm]
        % dpini     : B1 phase offset in addition to the one provided [rad]
        model_params    = { 'S0';   'MWF';  'IWF'; 'R1IEW'; 'kIEWM'; 'R2sMW';'R2sIW';'R2sEW'; 'freqMW';'freqIW';'dfreqBKG';'dpini'};
        ub              = [    2;     0.3;      1;       2;      10;     300;     40;     40;     0.25;    0.05;       0.4;   pi/2];
        lb              = [    0;       0;      0;    0.25;       0;      40;      2;      2;    -0.05;    -0.1;      -0.4;  -pi/2];
        startpoint      = [    1;     0.1;    0.8;       1;       0;     100;     15;     21;     0.04;       0;         0;      0];

    end

    properties (GetAccess = public, SetAccess = protected)

        x_i     = -0.1;         % ppm
        x_a     = -0.1;         % ppm
        E       = 0.02;         % ppm
        rho_mw  = 0.36/0.86;    % ratio
        t1_mw   = 234e-3;       % s

        % hardware setting
        B0      = 3; % T
        B0dir   = [0;0;1]; % main magnetic field direction with respect to FOV

        thres_R2star    = 100;  % 1/s, upper bound
        thres_T1        = 3;    % s, upper bound

        te;
        tr;
        fa;
        
    end
    
    methods

        % constructuor
        function this = gpuMCRMWI(te,tr,fa,fixed_params)
        % MCR-MWI
        % obj = gpuNEXI(b, Delta, Nav)
        %
        % Input
        % ----------
        % te        : Echo time [s]
        % tr        : repetition time [s]
        % fa        : flip angle [degree]
        % fixed_params: parameter to be fixed
        %       - x_i   : isotropic susceptibility of myelin [ppm]
        %       - x_a   : anisotropic susceptibility of myelin [ppm]
        %       - E     : exchange induced frequency shift [ppm]
        %       - rho_mw: myelin water proton ratio
        %       - B0    : main magnetic field strength [T]
        %       - B0dir : main magnetic field direction, [x,y,z]
        %       - t1_mw : myelin (water) T1 [s]
        %
        % Output
        % ----------
        % this      : object of a fitting class
        %
        % Author:
        %  Kwok-Shing Chan (kchan2@mgh.harvard.edu) 
        %  Copyright (c) 2023 Massachusetts General Hospital
        %
            
            this.te = single(te(:));
            this.tr = single(tr(:));
            this.fa = single(fa(:));
            % fixed tissue and scanner parameters
            if nargin == 4
                if isfield(fixed_params,'x_i')
                    this.x_i     = single(fixed_params.x_i);
                end
                if isfield(fixed_params,'x_a')
                    this.x_a     = single(fixed_params.x_a);
                end
                if isfield(fixed_params,'E')
                    this.E       = single(fixed_params.E);
                end
                if isfield(fixed_params,'rho_mw')
                    this.rho_mw  = single(fixed_params.rho_mw);
                end
                if isfield(fixed_params,'B0')
                    this.B0      = single(fixed_params.B0);
                end
                if isfield(fixed_params,'B0dir')
                    this.B0dir   = single(fixed_params.B0dir);
                end
                if isfield(fixed_params,'t1_mw')
                    this.t1_mw   = single(fixed_params.t1_mw);
                end
                if isfield(fixed_params,'thres_R2s')
                    this.thres_R2star = single(fixed_params.thres_R2star);
                end
                if isfield(fixed_params,'thres_T1')
                    this.thres_T1   = single(fixed_params.thres_T1);
                end
            end
        end
        
        % update properties according to lmax
        function this = updateProperty(this, fitting)

            if fitting.isComplex == 0
                for kpar = {'dfreqBKG','dpini'}
                    idx = find(ismember(this.model_params,kpar));
                    this.model_params(idx)    = [];
                    this.lb(idx)              = [];
                    this.ub(idx)              = [];
                    this.startpoint(idx)      = [];
                end
            end

            % DIMWI
            if fitting.DIMWI.isFitFreqIW == 0
                idx = find(ismember(this.model_params,'freqIW'));
                this.model_params(idx)    = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startpoint(idx)      = [];
            end

            if fitting.DIMWI.isFitFreqMW == 0
                idx = find(ismember(this.model_params,'freqMW'));
                this.model_params(idx)    = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startpoint(idx)      = [];
            end

            if fitting.DIMWI.isFitIWF == 0
                idx = find(ismember(this.model_params,'IWF'));
                this.model_params(idx)    = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startpoint(idx)      = [];
            end

            if fitting.DIMWI.isFitR2sEW == 0
                idx = find(ismember(this.model_params,'R2sEW'));
                this.model_params(idx)    = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startpoint(idx)      = [];
            end

            if fitting.isFitExchange == 0
                idx = find(ismember(this.model_params,'kIEWM'));
                this.model_params(idx)    = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startpoint(idx)      = [];
            end

        end

        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('================================');
            disp('MCR-(DI)MWI with askAdam solver');
            disp('================================');
            
            disp('----------------')
            disp('Data Information');
            disp('----------------')
            disp([  'Field strength (T)                     : ' num2str(this.B0)]);
            fprintf('Echo time, TE (ms)                     : [%s] \n',num2str((this.te*1e3).',' %.2f'));
            fprintf('Repetition time, TR (ms)               : %s \n',num2str((this.tr*1e3).',' %.2f'));
            fprintf('Flip angles (degree)                   : [%s] \n',num2str((this.fa).',' %.2f'));
            
            disp('---------------------')
            disp('Parameter to be fixed')
            disp('---------------------')
            disp(['Relative myelin water density            : ' num2str(this.rho_mw)]);
            disp(['Myelin isotropic susceptibility (ppm)    : ' num2str(this.x_i)]);
            disp(['Myelin anisotropic susceptibility (ppm)  : ' num2str(this.x_a)]);
            disp(['Exchange term (ppm)                      : ' num2str(this.E)]);
            disp(['Myelin T1 (ms)                           : ' num2str(this.t1_mw*1e3)]);

            fprintf('\n')

        end

        %% higher-level data fitting functions
        % Wrapper function of fit to handle image data; automatically segment data and fitting in case the data cannot fit in the GPU in one go
        function  [out] = estimate(this, data, mask, extraData, fitting)
        % Perform MCR-MWI model parameter estimation based on askAdam
        % Input data are expected in multi-dimensional image
        % 
        % Input
        % -----------
        % data      : 5D variable flip angle, multiecho GRE, [x,y,z,te,fa]
        % mask      : 3D signal mask, [x,y,z]
        % extradata : Optional additional data
        %   .freqBKG: 3D/4D initial estimation of total field [Hz] (highly recommended)
        %   .pini   : 3D initial estimation of B1 offset [rad]  (highly recommended)
        %   .ff     : 3D/4D fibre fraction map, [x,y,z,nF] (for GRE-DIMWI only)
        %   .theta  : 3D/4D angle between B0 and fibre orientation, [x,y,z, nF] (for GRE-DIMWI only)
        %   .IWF    : 3D volume fractino IC/(IC+EC), [x,y,z] (for GRE-DIMWI only)
        % fitting   : fitting algorithm parameters (see fit function)
        % 
        % Output
        % -----------
        % out       : output structure contains all estimation results
        % 
            
           % display basic info
            this.display_data_model_info;

            % get all fitting algorithm parameters 
            fitting = this.check_set_default(fitting);
            if isreal(data)
                fitting.isComplex = false;
            else
                fitting.isComplex = true;
            end

            % determine fitting parameters
            this = this.updateProperty(fitting);

            % make sure input data are valid
            [extraData,mask] = this.validate_data(data,extraData,mask,fitting);

            % compute rotationally invariant signal if needed
            [data, scaleFactor] = this.prepare_data(data,mask, extraData.b1);

            % mask sure no nan or inf
            [data,mask] = askadam.remove_img_naninf(data,mask);

            % get matrix size
            dims = size(data);

            % convert datatype to single
            data    = single(data);
            mask    = mask > 0;

            % determine if we need to divide the data to fit in GPU
            g = gpuDevice; reset(g);
            memoryFixPerVoxel       = 0.0013;   % get this number based on mdl fit
            memoryDynamicPerVoxel   = 0.05;     % get this number based on mdl fit
            [NSegment,maxSlice]     = askadam.find_optimal_divide(mask,memoryFixPerVoxel,memoryDynamicPerVoxel);

            % parameter estimation
            out = [];
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
                dwi_tmp     = data(:,:,slice,:,:);
                mask_tmp    = mask(:,:,slice);
                fields      = fieldnames(extraData);
                for kfield = 1:numel(fields); extraData_tmp.(fields{kfield}) = extraData.(fields{kfield})(:,:,slice,:,:); end

                % run fitting
                [out_tmp]    = this.fit(dwi_tmp,mask_tmp,fitting,extraData_tmp);

                % restore 'out' structure from segment
                out = askadam.restore_segment_structure(out,out_tmp,slice,ks);

            end
            out.mask = mask;
            % rescale S0
            out.final.S0    = out.final.S0 *scaleFactor;
            out.min.S0      = out.min.S0 *scaleFactor;

            % save the estimation results if the output filename is provided
            askadam.save_askadam_output(fitting.output_filename,out)

        end

        % Data fitting function, can be 2D (voxel) or 4D (image-based)
        function [out] = fit(this,data,mask,fitting,extraData)
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
        %   .lossFunction       : loss for data fidelity term, 'L1'|'L2'|'MSE', default = 'L1'
        %   .display            : online display the fitting process on figure, true|false, defualt = false
        %   .isWeighted         : is cost function weighted, true|false, default = true
        %   .weightMethod       : Weighting method, '1stecho'|'norm', default = '1stecho'
        %   .weightPower        : power order of the weight, default = 2
        %   .DIMWI.isFitIWF     : Vic is a free parameter, default = true
        %   .DIMWI.isFitFreqMW  : MW frequency is a free parameter, default = true
        %   .DIMWI.isFitFreqIW  : IW frequency is a free parameter, default = true
        %   .DIMWI.isFitR2sEW   : EW R2* is a free parameter, default = true
        %   .isFitExchange      : exchange is a free parameter, default = true
        %   .isEPG              : use EPG-X signal instead of BM analytical solution, default = true
        % 
        % Output
        % -----------
        % out       : output structure
        %   .final      : final results (see properties for other parameters)
        %       .loss       : final loss metric
        %   .min        : results with the minimum loss metric across all iterations
        %       .loss       : loss metric      
        %
        % Description: askAdam Image-based NEXI model fitting
        %
        % Kwok-Shing Chan @ MGH
        % kchan2@mgh.harvard.edu
        % Date created: 19 July 2024
        % Date modified:
        %
        %

            % load pre-trained ANN
            ann_epgx_phase = load('MCRMWI_MLP_EPGX_leakyrelu_N2e6_phase_v2.mat','dlnet');
            ann_epgx_phase.dlnet.alpha = 0.01;
            ann_epgx_magn  = load('MCRMWI_MLP_EPGX_leakyrelu_N2e6_magn.mat','dlnet');
            ann_epgx_magn.dlnet.alpha = 0.01;
            
            % check GPU
            g = gpuDevice;
            
            % get image size
            dims = size(data,1:3);

            %%%%%%%%%%%%%%%%%%%% 1. Validate and parse input %%%%%%%%%%%%%%%%%%%%
            if nargin < 3 || isempty(mask)
                % if no mask input then fit everthing
                mask = ones(dims);
            end

            if nargin < 4
                fitting = struct();
            end

            % set initial starting points
            pars0 = this.estimate_prior(data,mask,extraData);

            % get all fitting algorithm parameters 
            fitting                 = this.check_set_default(fitting);
            fitting.model_params    = this.model_params;
            fitting.Nsample         = numel(mask(mask ~= 0)); 

            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(this.model_params)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(this.model_params)); end
            
            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w = this.compute_optimisation_weights(data,fitting,mask); % This is a customised funtion
            w = dlarray(gpuArray(single(w)).','CB');

            % put data input gpuArray
            if fitting.isComplex
                % data = gpuArray(single(cat(5,real(data),imag(data))));
                data = gpuArray(single(cat(1,this.vectorise_NDto2D(real(data),mask).',this.vectorise_NDto2D(imag(data),mask).')));
            else
                % data = gpuArray(single(data));
                data = gpuArray(single(this.vectorise_NDto2D(data,mask).'));
            end
            mask = gpuArray(logical(mask));
            fields = fieldnames(extraData); for kfield = 1:numel(fields); extraData.(fields{kfield}) = dlarray(gpuArray( this.vectorise_NDto2D(extraData.(fields{kfield}),mask).')); end

            % 2.2 display optimisation algorithm parameters
            askadam.display_basic_fitting_parameters(fitting);
            this.display_algorithm_info(fitting)
            
            % 2.3 askAdam optimisation main
            askadamObj = askadam();
            out = askadamObj.optimisation(pars0, @this.modelGradients, data, mask, w, fitting, extraData, ann_epgx_phase.dlnet, ann_epgx_magn.dlnet);

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            disp('The process is completed.')
            
            % clear GPU
            reset(g)
            
        end

        % compute the gradient and loss of forward modelling
        function [gradients,loss,loss_fidelity,loss_reg] = modelGradients(this, parameters, data, mask, weights, fitting, extraData, dlnet_phase, dlnet_magn)
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
            askadamObj = askadam();
            % parameters = askadamObj.rescale_parameters(parameters,fitting.lb,fitting.ub,fitting.model_params);

            % Masking
            % 1st dimension: TE [real;imag]
            % data = this.vectorise_5Dto2D(data,mask).';
            
            % Forward model
            [Rreal, Rimag] = this.FWD(askadamObj.rescale_parameters(parameters,fitting.lb,fitting.ub,fitting.model_params),fitting,extraData,mask, dlnet_phase, dlnet_magn);
            R = cat(1,Rreal,Rimag); R(isinf(R)) = 0; R(isnan(R)) = 0;

            % Masking
            R       = dlarray(R(:).',       'CB');
            data    = dlarray(data(:).',    'CB');

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
                cost        = askadam.reg_TV(squeeze(parameters.(fitting.regmap)),mask,fitting.TVmode,fitting.voxelSize);
                loss_reg    = sum(abs(cost),"all")/fitting.Nsample *fitting.lambda;
            else
                loss_reg = 0;
            end
            
            % compute loss
            loss = loss_fidelity + loss_reg;
            
            % Calculate gradients with respect to the learnable parameters.
            gradients = dlgradient(loss,parameters);
        
        end
        
        % compute weights for optimisation
        function w = compute_optimisation_weights(this,data,fitting,mask)
        % 
        % Output
        % ------
        % w         : 1D signal masked wegiths
        %
            if fitting.isWeighted
                switch lower(fitting.weightMethod)
                    case 'norm'
                       % weights using echo intensity, as suggested in Nam's paper
                        w = sqrt(abs(data));
                    case '1stecho'
                        p = fitting.weightPower;
                        % weights using the 1st echo intensity of each flip angle
                        w = bsxfun(@rdivide,abs(data).^p,abs(data(:,:,:,1,:)).^p);
                end
            else
                w = ones(size(data));
            end

            w(w>1) = 1; w(w<0) = 0;
            w = this.vectorise_NDto2D(w,mask).';

            if fitting.isComplex
                w = repmat(w,2,1);
            end
            w = w(:);
        end

        %% Prior estimation related functions

        % using maximum likelihood method to estimate starting points
        function pars0 = estimate_prior(this,data,mask,extraData)
        % Estimation starting points 

            dims = size(data,1:3);

            for k = 1:numel(this.model_params)
                pars0.(this.model_params{k}) = single(this.startpoint(k)*ones(dims));
            end
            
            disp('Estimate starting points based on hybrid fixed points/prior information ...')
            
            % S0
            S0 = zeros([dims numel(this.fa)]);
            for kfa = 1:numel(this.fa)
                [~,S0(:,:,:,kfa)]  = this.R2star_trapezoidal(abs(data(:,:,:,:,kfa)),this.te);
            end
            S0(isnan(S0)) = 0; S0(isinf(S0)) = 0;
            despot1_obj = despot1(this.tr,this.fa);
            [t1, M0, ~] = despot1_obj.estimate(S0, mask, extraData.b1);
            pars0.(this.model_params{1}) = single(M0);

            % R1
            pars0.R1IEW = single(1./t1); pars0.R1IEW(isnan(pars0.R1IEW)) = 0; pars0.R1IEW(isinf(pars0.R1IEW)) = 0;

            % freqBKG
            pars0.dfreqBKG = repmat(pars0.dfreqBKG,[1,1,1,numel(this.fa)]);

            [R2s,~]  = this.R2star_trapezoidal(mean(abs(data),5),this.te);
            % R2*IW
            idx = find(ismember(this.model_params,'R2sIW'));
            R2sIW = R2s - 3;
            R2sIW(isnan(R2sIW)) = single(this.startpoint(idx)); R2sIW(isinf(R2sIW)) = single(this.startpoint(idx)); 
            R2sIW(R2sIW < this.lb(idx)) = single(this.lb(idx)); R2sIW(R2sIW > this.ub(idx)) = single(this.ub(idx));
            pars0.R2sIW = R2sIW;

            % R2*EW
            idx = find(ismember(this.model_params,'R2sEW'));
            if ~isempty(idx)
                R2sEW = R2s + 3;
                R2sEW(isnan(R2sEW)) = single(this.startpoint(idx)); R2sEW(isinf(R2sEW)) = single(this.startpoint(idx)); 
                R2sEW(R2sEW < this.lb(idx)) = single(this.lb(idx)); R2sEW(R2sEW > this.ub(idx)) = single(this.ub(idx));
                pars0.R2sEW = R2sEW;
            end

            % [~,mwf] = this.superfast_mwi_2m_standard(abs(data),this.te,[]);
            % mwf(mwf>0.1)    = 0.1;    % mainly WM
            % mwf(mwf<0.05)   = 0.02;   % mainly GM
            % mwf(R2s0>35)    = 0.02;
            % mwf             = smooth3(mwf);
            % mwf = single(this.startpoint(2)*ones(dims));
            % mwf(R2s0<15) = 0.05;
            % pars0.(this.model_params{2}) = single(mwf);
            
        end

        %% Signal related functions

        % compute the forward model
        function [Sreal, Simag] = FWD(this, pars, fitting, extraData, mask, dlnet_phase,dlnet_magn)

            nFA = numel(this.fa);

        % Forward model to generate NEXI signal
            if isempty(mask)
                S0   = pars.S0;
                mwf  = pars.MWF;
                if fitting.DIMWI.isFitIWF
                    iwf  = pars.IWF;
                else
                    iwf = extraData.IWF;
                end
                r2sMW   = pars.R2sMW;
                r2sIW   = pars.R2sIW;
                r1iew   = pars.R1IEW;

                if fitting.isFitExchange
                    kiewm = pars.kIEWM;
                end
    
                if fitting.DIMWI.isFitR2sEW
                    r2sEW  = pars.R2sEW;
                end
                if fitting.DIMWI.isFitFreqMW
                    freqMW  = pars.freqMW;
                end
                if fitting.DIMWI.isFitFreqIW
                    freqIW  = pars.freqIW;
                end
                % external effects
                if ~fitting.isComplex % magnitude fitting
                    freqBKG = 0;                          
                    pini    = 0;
                else    % other fittings
                    freqBKG = pars.dfreqBKG + extraData.freqBKG; 
                    pini    = pars.dpini + extraData.pini;
                end

                % move fibre oreintation to the 5th dimension
                extraData.ff    = permute(extraData.ff,[1 2 3 5 4]);
                extraData.theta = permute(extraData.theta,[1 2 3 5 4]);

                % TE = permute(this.te,[2 3 4 1]);
                TE = gpuArray(dlarray( permute(this.te,[2 3 4 1]) ));
                FA = gpuArray(dlarray( deg2rad(this.fa) ));
                % FA = gpuArray(dlarray( permute(deg2rad(this.fa), [2 3 4 5 1]) ));
    
            else
                % mask out voxels to reduce memory
                S0   = squeeze(pars.S0(mask)).';
                mwf  = squeeze(pars.MWF(mask)).';
                if fitting.DIMWI.isFitIWF
                    iwf  = squeeze(pars.IWF(mask)).';
                else
                    iwf = extraData.IWF;
                end
                r2sMW   = squeeze(pars.R2sMW(mask)).';
                r2sIW   = squeeze(pars.R2sIW(mask)).';
                r1iew   = squeeze(pars.R1IEW(mask)).';

                if fitting.isFitExchange
                    kiewm = squeeze(pars.kIEWM(mask)).';
                end
    
                if fitting.DIMWI.isFitR2sEW
                    r2sEW  = squeeze(pars.R2sEW(mask)).';
                end
                if fitting.DIMWI.isFitFreqMW
                    freqMW  = squeeze(pars.freqMW(mask)).';
                end
                if fitting.DIMWI.isFitFreqIW
                    freqIW  = squeeze(pars.freqIW(mask)).';
                end
                % external effects
                if ~fitting.isComplex % magnitude fitting
                    freqBKG = 0;                          
                    pini    = 0;
                else    % other fittings
                    if isscalar(mask)
                        dshift = -2;
                    else
                        dshift = -1;
                    end
                    freqBKG = shiftdim((this.vectorise_NDto2D(pars.dfreqBKG,mask).' + extraData.freqBKG).',dshift) ; 
                    pini    = squeeze(pars.dpini(mask)).' + extraData.pini;
                end

                extraData.ff    = permute(extraData.ff,[3 2 4 1]);
                extraData.theta = permute(extraData.theta,[3 2 4 1]);

                TE = gpuArray(dlarray( this.te ));
                FA = gpuArray(dlarray( deg2rad(this.fa) ));
            end
                
            % Forward model
            %%%%%%%%%%%%%%%%%%%% DIMWI related operations %%%%%%%%%%%%%%%%%%%%
            % Compartmental Signals
            S0MW        = S0 .* mwf;
            S0IW        = S0 .* (1-mwf) .* iwf;
            S0EW        = S0 .* (1-mwf) .* (1-iwf);
            totalVolume = S0IW + S0EW + S0MW/this.rho_mw;
            MVF         = (S0MW/this.rho_mw) ./ totalVolume;

            if ~fitting.DIMWI.isFitFreqMW || ~fitting.DIMWI.isFitFreqIW || ~fitting.DIMWI.isFitR2sEW
                hcfm_obj = HCFM(this.te,this.B0);

                % derive g-ratio 
                g = hcfm_obj.gratio(abs(S0IW),abs(S0MW)/this.rho_mw);

            end
            
           % extra decay on extracellular water estimated by HCFM 
           if ~fitting.DIMWI.isFitR2sEW
                
                % assume extracellular water has the same T2* as intra-axonal water
                r2sEW   = r2sIW;
                % fibre volume fraction
                fvf     = hcfm_obj.FibreVolumeFraction(abs(S0IW),abs(S0EW),abs(S0MW)/this.rho_mw);

                % signal dephase in extracellular water due to myelin sheath, Eq.[A7]
                decayEW = hcfm_obj.DephasingExtraaxonal(fvf,g,this.x_i,this.x_a,extraData.theta);

           else
                decayEW = 0;
           end

           % determine the source of compartmental frequency shifts
            if ~fitting.DIMWI.isFitFreqMW || ~fitting.DIMWI.isFitFreqIW
               
                % compute frequency shifts given theta
                if ~fitting.DIMWI.isFitFreqMW 

                    freqMW = hcfm_obj.FrequencyMyelin(this.x_i,this.x_a,g,extraData.theta,this.E) / (this.B0*this.gyro);

                end
                if ~fitting.DIMWI.isFitFreqIW 

                    freqIW = hcfm_obj.FrequencyAxon(this.x_a,g,extraData.theta) / (this.B0*this.gyro);

                end
            end

            freqEW = 0;

            %%%%%%%%%%%%%%%%%%%% T1 model %%%%%%%%%%%%%%%%%%%%
            
            if fitting.isFitExchange
                if fitting.isEPG
                    % EPG-X
                    features = feature_preprocess_MCRMWI_MLP_EPGX_leakyrelu( repmat(MVF(:),nFA,1),      1./repmat(r1iew(:),nFA,1), ...
                                                                             repmat(kiewm(:),nFA,1),    (squeeze(FA) .* extraData.b1(:).').', ...    % true_famp = (FA .* extraData.b1(:).').';
                                                                             this.tr, 1./repmat(r2sIW(:),nFA,1), this.t1_mw);
                    features    = gpuArray( dlarray(features,'CB'));
                    
                    % phase of long T2 components
                    S0IEW_phase   = mlp_model_leakyRelu(dlnet_phase.parameters,features,dlnet_phase.alpha);
                    S0IEW_phase   = reshape(S0IEW_phase,[size(MVF),nFA]);
    
                    % signal_steadystate_phase   = reshape(signal_steadystate_phase,size(mvf,1),size(mvf,2),size(mvf,3),1,nFA);
                    Ss_diff    = mlp_model_leakyRelu(dlnet_magn.parameters,features,dlnet_magn.alpha);
                    Ss_diff    = shiftdim( reshape(Ss_diff,[2, size(MVF), nFA]), 1);
    
                    % true_famp = permute(FA,[2:ndims(MVF)+1 1]) .* extraData.b1;
                    % [S0MW, S0IEW]   = (this.model_BM_2T1(this.tr, shiftdim(FA,-ndims(MVF)) .* extraData.b1, MVF,r1iew,1./this.t1_mw,kiewm));   % S0MW here is indeed S0Myelin, recycle variable to reduce memory
                    [S0MW, S0IEW]   = (this.model_BM_2T1_analytical(this.tr, shiftdim(FA,-ndims(MVF)) .* extraData.b1, MVF,r1iew,1./this.t1_mw,kiewm));   % S0MW here is indeed S0Myelin, recycle variable to reduce memory
                    S0MW  = (S0MW  + Ss_diff(:,:,:,1)) .* totalVolume .* this.rho_mw;
                    S0IEW = (S0IEW + Ss_diff(:,:,:,2)) .* totalVolume;
                    % S0_mag          = (this.model_BM_2T1(this.tr, shiftdim(FA,-ndims(MVF)) .* extraData.b1, MVF,r1iew,1./this.t1_mw,kiewm) + Ss_diff ) .* totalVolume;
                
                else
                    % BM analytical solution
                    % S0_mag          = (this.model_BM_2T1(this.tr, shiftdim(FA,-ndims(MVF)) .* extraData.b1, MVF,r1iew,1./this.t1_mw,kiewm)) .* totalVolume;
                    % [S0MW, S0IEW]   = (this.model_BM_2T1(this.tr, shiftdim(FA,-ndims(MVF)) .* extraData.b1, MVF,r1iew,1./this.t1_mw,kiewm));   % S0MW here is indeed S0Myelin, recycle variable to reduce memory
                    [S0MW, S0IEW]   = (this.model_BM_2T1_analytical(this.tr, shiftdim(FA,-ndims(MVF)) .* extraData.b1, MVF,r1iew,1./this.t1_mw,kiewm));   % S0MW here is indeed S0Myelin, recycle variable to reduce memory
                    S0MW            = S0MW  .* totalVolume .* this.rho_mw;
                    S0IEW           = S0IEW .* totalVolume;
                    S0IEW_phase     = 0;
                end
            else
                [S0MW, S0IEW] = this.model_Bloch_2T1(this.tr,S0MW,S0IW+S0EW,this.t1_mw,1./r1iew,shiftdim(FA,-ndims(MVF)) .* extraData.b1);
                S0IEW_phase   = 0;
            end

            %%%%%%%%%%%%%%%%%%%% Forward model %%%%%%%%%%%%%%%%%%%%
            if isempty(mask)
                % Image-based operation
                if ndims(pars.MWF) == 2
                    S0MW = permute(S0MW,[1 2 5 4 3]);
                    S0IW = permute(S0IW,[1 2 5 4 3]);
                    S0EW = permute(S0EW,[1 2 5 4 3]);
                    freqBKG = permute(freqBKG,[3 2 4 5 1]);
                end

                Sreal = sum((   S0MW .* exp(-TE .* r2sMW) .* cos(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0IW .* exp(-TE .* r2sIW) .* cos(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0EW .* exp(-TE .* r2sEW) .* cos(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini) .* exp(-decayEW) ).*extraData.ff,6);

                Simag = sum((   S0MW .* exp(-TE .* r2sMW) .* sin(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0IW .* exp(-TE .* r2sIW) .* sin(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0EW .* exp(-TE .* r2sEW) .* sin(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini) .* exp(-decayEW) ).*extraData.ff,6);


            else
                % voxel-based operation
                Sreal = sum((   S0MW            .* exp(-TE .* r2sMW) .* cos(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...               % MW
                                S0IEW.*iwf      .* exp(-TE .* r2sIW) .* cos(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini + S0IEW_phase) + ...    % IW
                                S0IEW.*(1-iwf)  .* exp(-TE .* r2sEW) .* cos(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini + S0IEW_phase) .* exp(-decayEW) ).*extraData.ff,4);
    
                Simag = sum((   S0MW            .* exp(-TE .* r2sMW) .* sin(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0IEW.*iwf      .* exp(-TE .* r2sIW) .* sin(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini + S0IEW_phase) + ...
                                S0IEW.*(1-iwf)  .* exp(-TE .* r2sEW) .* sin(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini + S0IEW_phase) .* exp(-decayEW) ).*extraData.ff,4);

                if ~fitting.isComplex
                    Sreal = permute( sqrt(Sreal.^2 + Simag.^2), [1 3 2]);
                    Simag = 0;
                    
                else
                    Sreal = reshape( permute( Sreal, [1 3 2]), numel(TE)*numel(FA), numel(MVF));
                    Simag = reshape( permute( Simag, [1 3 2]), numel(TE)*numel(FA), numel(MVF));

                end
            end
    
        end
        
        %% Utilities

        % validate extra data
        function [extraData,mask] = validate_data(this,data,extraData,mask,fitting)

           % % check if the signal is monotonic decay
           %  [~,I] = max(abs(data),[],4);
           %  mask = and(mask,min(I<4,[],5));

            dims = size(data,1:3);

            if ~fitting.DIMWI.isFitIWF && ~isfield(extraData,'IWF')
                error('Field IWF is missing in exraData structure variable for DIMWI model');
            end
            if (~fitting.DIMWI.isFitFreqMW || ~fitting.DIMWI.isFitFreqIW || ~fitting.DIMWI.isFitR2sEW) && ~isfield(extraData,'theta')
                error('Field theta is missing in exraData structure variable for DIMWI model');
            end
            
            if ~isfield(extraData,'ff') || (fitting.DIMWI.isFitFreqMW && fitting.DIMWI.isFitFreqIW && fitting.DIMWI.isFitR2sEW) 
                extraData.ff = ones(dims);
            else
                % normalise fibre fraction
                extraData.ff                        = bsxfun(@rdivide,extraData.ff,sum(extraData.ff,4));
                mask = and(mask,min(~isnan(extraData.ff),[],4));
                extraData.ff(isnan(extraData.ff))   = 0;

            end
            if ~isfield(extraData,'theta') || (fitting.DIMWI.isFitFreqMW && fitting.DIMWI.isFitFreqIW && fitting.DIMWI.isFitR2sEW) 
                extraData.theta = zeros(dims);
            end
            if ~isfield(extraData,'freqBKG')
                extraData.freqBKG = zeros(dims);
                if fitting.isComplex
                    warning('No total field map is provided for fitting complex-valued data.');
                end
            end
            if ~isfield(extraData,'pini')
                % extraData.pini = zeros(dims);
                extraData.pini = angle( data(:,:,:,1,1) ./ exp(1i* 2*pi*extraData.freqBKG * (this.B0*this.gyro) .* permute(this.te(1),[2 3 4 1])));
            end
            if ~isfield(extraData,'b1')
                extraData.b1 = ones(dims);
                warning('Missing B1+ map. Assuming B1=1 everywhere.');
            end

            if size(extraData.theta,4) ~= size(extraData.ff,4)
                error('The last dimention of the theta map does not match the last dimension of the fibre fraction map');
            end

            fields = fieldnames(extraData); for kfield = 1:numel(fields); extraData.(fields{kfield}) = single( extraData.(fields{kfield})); end

            despot1_obj          = despot1(this.tr,this.fa);
            [t1, ~, mask_fitted] = despot1_obj.estimate(permute(abs(data(:,:,:,1,:)),[1 2 3 5 4]), mask, extraData.b1);
            [R2star,~]           = R2star_trapezoidal(mean(abs(data),5),this.te);

            mask = and(mask,and(t1<=this.thres_T1,mask_fitted));
            mask = and(mask,R2star<=this.thres_R2star);
            mask = and(mask,min(abs(data(:,:,:,1,:))>0,[],5));
            
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
       
        %% signal

        % Bloch-McConnell 2-pool exchange steady-state signal based on analytical solution
        function [S0M, S0IEW] = model_BM_2T1_analytical(TR,true_famp,f,R1f,R1r,kfr)

            a = (-R1f-kfr);
            b = (1-f).*kfr./f;
            c = kfr;
            d = -R1r-b;
            
            % Eigenvalues
            lambda1 = (((a+d) + sqrt((a+d).^2-4.*(a.*d-b.*c)))./2); 
            lambda2 = (((a+d) - sqrt((a+d).^2-4.*(a.*d-b.*c)))./2); 
            
            ELambda1TR = exp(lambda1.*TR);
            ELambda2TR = exp(lambda2.*TR);
            
            % exponential matrix elements
            e11 = (ELambda1TR.*(a-lambda2) - ELambda2TR.*(a-lambda1)) ./ (lambda1 - lambda2);
            e12 = b.*(ELambda1TR-ELambda2TR)./ (lambda1 - lambda2);
            e21 = c.*(ELambda1TR-ELambda2TR)./ (lambda1 - lambda2);
            e22 = (ELambda1TR.*(d-lambda2) - ELambda2TR.*(d-lambda1)) ./ (lambda1 - lambda2);
            
            % constant term
            C = sin(true_famp)./((1-(e22+e11).*cos(true_famp)+(e11.*e22-e21.*e12).*cos(true_famp).^2).*(a.*d-b.*c));
            
            ap = ((1-e22.*cos(true_famp)).*(e11-1) + e12.*e21.*cos(true_famp));
            bp = ((1-e22.*cos(true_famp)).*e12 + (e12.*cos(true_famp)).*(e22-1));
            cp = (e21.*cos(true_famp).*(e11-1) + (1-e11.*cos(true_famp)).*e21) ;
            dp = (e21.*cos(true_famp).*e12 + (1-e11.*cos(true_famp)).*(e22-1));
            
            S0IEW   = ((ap.*d-bp.*c).*R1f.*(1-f) + (-ap.*b+bp.*a).*R1r.*f).*C;
            S0M     = ((cp.*d-c.*dp).*R1f.*(1-f) + (-cp.*b+a.*dp).*R1r.*f).*C;
        
        end

        % Bloch-McConnell 2-pool exchange steady-state signal
        function [S0M, S0IEW] = model_BM_2T1(TR,true_famp,f,R1f,R1r,kfr)
        % Matlab symbolic toolbox
        % f : myelin volume fraction
        % R1f : R1 of free (IE) water
        % kfr : exchange rate from free (IE) water to myelin
        % true_famp: true flip angle map in radian (5D), flip angle in 5th dim
        % ss_pool: 1st-3rd dim:image; 4th dim: TE;5th dim: FA;6th dim: [S_M,S_IEW]
        
            % free pool
            S0IEW = ((f./2 - 1./2).*(kfr.*sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f - ...
                        (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + ...
                        R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) - ...
                        2.*sin(2.*true_famp).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
                        2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) + ...
                        2.*kfr.*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
                        2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp) + ...
                        sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
                        2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*(R1f.^2.*f.^2 - ...
                        2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) + ...
                        2.*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
                        4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
                        4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) - ...
                        2.*kfr.*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + ...
                        R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp) - ...
                        kfr.*sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
                        4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) - ...
                        4.*exp((TR.*(kfr + R1f.*f + R1r.*f))./f).*sin(true_famp).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
                        2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) + 2.*exp((TR.*(kfr + R1f.*f + ...
                        R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + ...
                        2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
                        2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) + ...
                        sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
                        2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*(R1f.^2.*f.^2 - ...
                        2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) - ...
                        R1f.*f.*sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + ...
                        R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) + R1r.*f.*sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + ...
                        R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + ...
                        kfr.^2).^(1./2)))./(2.*f)) - 2.*R1f.*f.*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
                        2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp) + ...
                        2.*R1r.*f.*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
                        2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp) + ...
                        2.*R1f.*f.*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + ...
                        R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp) + ...
                        R1f.*f.*sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
                        2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) - ...
                        2.*R1r.*f.*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + ...
                        R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp) - ...
                        R1r.*f.*sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
                        2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f))))./(2.*(exp((TR.*(kfr + R1f.*f + ...
                        R1r.*f))./f) + cos(true_famp).^2 - exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
                        2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*cos(true_famp) - exp((TR.*(kfr + ...
                        R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + ...
                        2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*cos(true_famp)).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2));
            
            % restricted pool
            S0M = -(2.*f.*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + ...
                        R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
                        4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) - 2.*f.*sin(2.*true_famp).*(R1f.^2.*f.^2 - ...
                        2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) - ...
                        2.*f.*kfr.*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
                        4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp) - f.*kfr.*sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f + ...
                        (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) - ...
                        4.*f.*exp((TR.*(kfr + R1f.*f + R1r.*f))./f).*sin(true_famp).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
                        4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) - 2.*R1f.*f.^2.*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
                        4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp) - ...
                        R1f.*f.^2.*sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
                        4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) + 2.*R1r.*f.^2.*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
                        4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp) + ...
                        R1r.*f.^2.*sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
                        4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) + 2.*f.*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
                        4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp).*(R1f.^2.*f.^2 - ...
                        2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) + ...
                        f.*sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
                        4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
                        4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) + f.*kfr.*sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
                        4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) + 2.*f.*kfr.*exp((TR.*(kfr + R1f.*f + ...
                        R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + ...
                        2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp) + R1f.*f.^2.*sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - ...
                        2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) - ...
                        R1r.*f.^2.*sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
                        4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) + f.*sin(2.*true_famp).*exp((TR.*(kfr + R1f.*f + R1r.*f - ...
                        (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + ...
                        2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
                        4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) + 2.*R1f.*f.^2.*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
                        4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp) - ...
                        2.*R1r.*f.^2.*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
                        4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(true_famp))./(4.*(exp((TR.*(kfr + R1f.*f + R1r.*f))./f) + cos(true_famp).^2 - ...
                        exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + ...
                        2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*cos(true_famp) - exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
                        2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*cos(true_famp)).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
                        4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2));
             
            % idx = find(size(Sr) == 1);
            % if isempty(idx)
            %     idx = ndims(Sr) + 1;
            % end
            % 
            % ss_pool = cat(ndims(Sr)+1,Sr,Sf);
        
        end
        
        % Bloch non-exchanging 3-pool steady-state model
        function [S0MW, S0IEW] = model_Bloch_2T1(TR,M0MW,M0IEW,T1MW,T1IEW,true_famp)
        % s     : non-exchange steady-state signal, 1st dim: pool; 2nd dim: flip angles
        %
            S0MW    = M0MW  .* sin(true_famp) .* (1-exp(-TR./T1MW)) ./(1-cos(true_famp).*exp(-TR./T1MW));
            S0IEW   = M0IEW .* sin(true_famp) .* (1-exp(-TR./T1IEW))./(1-cos(true_famp).*exp(-TR./T1IEW));
    
        end

        % closed form single compartment solution
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

            % check weighted sum of cost function
            if ~isfield(fitting,'isWeighted')
                fitting2.isWeighted = true;
            end
            if ~isfield(fitting,'weightMethod')
                fitting2.weightMethod = '1stecho';
            end
            if ~isfield(fitting,'weightPower')
                fitting2.weightPower = 2;
            end
            
            % check hollow cylinder fibre model parameters
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitFreqMW')
                fitting2.DIMWI.isFitFreqMW = true;
            end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitFreqIW')
                fitting2.DIMWI.isFitFreqIW = true;
            end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitR2sEW')
                fitting2.DIMWI.isFitR2sEW = true;
            end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitIWF')
                fitting2.DIMWI.isFitIWF = true;
            end

            if ~isfield(fitting,'isFitExchange')
                fitting2.isFitExchange = true;
            end
            if ~isfield(fitting,'isEPG')
                fitting2.isEPG = true;
            end

            % get customised fitting setting check
            if ~isfield(fitting,'regmap')
                fitting2.regmap = 'MWF';
            end

        end

        function display_algorithm_info(fitting)
            %%%%%%%%%% 3. display some algorithm parameters %%%%%%%%%%
            disp('--------------');
            disp('Fitting option');
            disp('--------------');
            % type of fitting
            if fitting.isComplex
                disp('Fitting with complex-valued data');
            else 
                disp('Fitting with magnitude data');
            end

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

            disp('----------');
            disp('T1 options');
            disp('----------');
            if ~fitting.isFitExchange
                disp('Fit exchange  : False');
            else
                disp('Fit exchange  : True');
            end
            if ~fitting.isEPG
                disp('Use EGP-X     : False');
            else
                disp('Use EGP-X     : True');
            end

            disp('------------------------------------');
            disp('Diffusion informed MWI model options');
            disp('------------------------------------');
            if ~fitting.DIMWI.isFitIWF
                disp('Fit intra-axonal volume fraction  : False');
            else
                disp('Fit intra-axonal volume fraction  : True');
            end
            if ~fitting.DIMWI.isFitFreqMW
                disp('Fit frequency - myelin water      : False');
            else
                disp('Fit frequency - myelin water      : True');
            end
            if ~fitting.DIMWI.isFitFreqIW
                disp('Fit frequency - intra-axonal water: False');
            else
                disp('Fit frequency - intra-axonal water: True');
            end
            if ~fitting.DIMWI.isFitR2sEW
                disp('Fit R2* - extra-cellular water    : False');
            else
                disp('Fit R2* - extra-cellular water    : True');
            end

            disp('------------------------------------');

        end

        % vectorise N-D image to 2D with the 1st dimension=spataial dimension and 2nd dimension=combine from 4th and onwards 
        function [data, mask_idx] = vectorise_NDto2D(data,mask)

            dims = size(data,[1 2 3]);

            if nargin < 2
                mask = ones(dims);
            end

             % vectorise data
            data        = reshape(data,prod(dims),prod(size(data,4:ndims(data))));
            mask_idx    = find(mask>0);
            data        = data(mask_idx,:);

            if ~isreal(data)
                data = cat(2,real(data),imag(data));
            end

        end

        function st = cell2str(cellStr)
            cellStr= cellfun(@(x){[x ',']},cellStr);  % Add ',' after each string.
            st = cat(2,cellStr{:});  % Convert to string
            st(end) = [];  % Remove last ','
        end

        function theta = AngleBetweenV1MapAndB0(v1,b0dir)
        %
        % Input
        % --------------
        % v1            : 4D fibre orientation map in vector form
        % b0dir         : 1D vector of B0 direction
        %
        % Output
        % --------------
        % theta         : 3D angle map, in rad
        %
        % Description:
        %
        % Kwok-shing Chan @ DCCN
        % k.chan@donders.ru.nl
        % Date created: 20 March 2019
        % Date last modified: 25 October 2019
        %
        %

            % replicate B0 direction to all voxels
            b0dirmap = permute(repmat(b0dir(:),1,size(v1,1),size(v1,2),size(v1,3)),[2 3 4 1]);
            % compute angle between B0 direction and fibre orientation
            theta = atan2(vecnorm(cross(v1,b0dirmap),2,4), dot(v1,b0dirmap,4));
            
            % make sure the angle is in range [0, pi/2]
            theta(theta> (pi/2)) = pi - theta(theta> (pi/2));
        
        end
    
    end
end