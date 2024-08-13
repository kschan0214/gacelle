classdef gpuGREMWI
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 22 July 2024
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
        model_params    = { 'S0';   'MWF';  'IWF';  'R2sMW';'R2sIW';'R2sEW'; 'freqMW';'freqIW';'dfreqBKG';'dpini'};
        ub              = [    2;     0.3;      1;      300;     40;     40;     0.25;    0.05;       0.4;   pi/2];
        lb              = [    0;       0;      0;       40;      2;      2;    -0.05;    -0.1;      -0.4;  -pi/2];
        startpoint      = [    1;     0.1;    0.8;      100;     15;     21;     0.04;       0;         0;      0];

    end

    properties (GetAccess = public, SetAccess = protected)

        B0      = 3;            % T
        x_i     = -0.1;         % ppm
        x_a     = -0.1;         % ppm
        E       = 0.02;         % ppm
        rho_mw  = 0.36/0.86;    % ratio
        B0dir   = [0;0;1];      % unit vector [x,y,z]

        thres_R2star = 2;

        te
        
    end
    
    methods

        % constructuor
        function this = gpuGREMWI(te,fixed_params)
        % GRE-MWI 3-pool model
        % this = gpuGREMWI(te,fixed_params)
        %
        % Input
        % ----------
        % te        : Echo time [s]
        % fixed_params: parameter to be fixed
        %       - x_i   : isotropic susceptibility of myelin [ppm]
        %       - x_a   : anisotropic susceptibility of myelin [ppm]
        %       - E     : exchange induced frequency shift [ppm]
        %       - rho_mw: myelin water proton ratio
        %       - B0    : main magnetic field strength [T]
        %       - B0dir : main magnetic field direction, [x,y,z]
        %       - thres_R2s : threshold of single compartment R2* for refine brain mask [1/s]
        %
        % Output
        % ----------
        % this      : object of a fitting class
        %
        % Author:
        %  Kwok-Shing Chan (kchan2@mgh.harvard.edu) 
        %  Copyright (c) 2023 Massachusetts General Hospital
            
            this.te     = single(te(:));
            % fixed tissue and scanner parameters
            if nargin == 2
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
                if isfield(fixed_params,'thres_R2s')
                    this.thres_R2star   = single(fixed_params.thres_R2star);
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

        end

        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('================================');
            disp('GRE-(DI)MWI with askAdam solver');
            disp('================================');
            
            disp('----------------')
            disp('Data Information');
            disp('----------------')
            disp([  'Field strength (T)                     : ' num2str(this.B0)]);
            fprintf('Echo time, TE (ms)                     : [%s] \n',num2str((this.te*1e3).',' %.2f'));
            
            disp('---------------------')
            disp('Parameter to be fixed')
            disp('---------------------')
            disp(['Relative myelin water density            : ' num2str(this.rho_mw)]);
            disp(['Myelin isotropic susceptibility (ppm)    : ' num2str(this.x_i)]);
            disp(['Myelin anisotropic susceptibility (ppm)  : ' num2str(this.x_a)]);
            disp(['Exchange term (ppm)                      : ' num2str(this.E)]);

            fprintf('\n')

        end

        %% higher-level data fitting functions
        % Wrapper function of fit to handle image data; automatically segment data and fitting in case the data cannot fit in the GPU in one go
        function  [out] = estimate(this, data, mask, extraData, fitting)
        % Perform GRE-MWI model parameter estimation based on askAdam
        % Input data are expected in multi-dimensional image
        % 
        % Input
        % -----------
        % data      : 4D multi-echo GRE, [x,y,z,te]
        % mask      : 3D signal mask, [x,y,z]
        % extradata : Optional additional data
        %   .freqBKG: 3D initial estimation of total field [Hz] (highly recommended)
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
            [data, scaleFactor] = this.prepare_data(data,mask);

            % mask sure no nan or inf
            [data,mask] = askadam.remove_img_naninf(data,mask);

            % get matrix size
            dims = size(data);

            % convert datatype to single
            data    = single(data);
            mask    = mask >0;

            % determine if we need to divide the data to fit in GPU
            g = gpuDevice; reset(g);
            memoryFixPerVoxel       = 0.0013;   % get this number based on mdl fit
            memoryDynamicPerVoxel   = 0.005;     % get this number based on mdl fit
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
                dwi_tmp     = data(:,:,slice,:);
                mask_tmp    = mask(:,:,slice);
                fields      = fieldnames(extraData); for kfield = 1:numel(fields); extraData_tmp.(fields{kfield}) = single(extraData.(fields{kfield})(:,:,slice,:,:)); end

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
            pars0 = this.estimate_prior(data);

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
                % data = gpuArray(single(cat(4,real(data),imag(data))));
                data = gpuArray(single(cat(1,this.vectorise_NDto2D(real(data),mask).',this.vectorise_NDto2D(imag(data),mask).')));
            else
                % data = gpuArray(single(data));
                data = gpuArray(single(this.vectorise_NDto2D(data,mask).'));
            end
            mask = gpuArray(logical(mask));
            fields = fieldnames(extraData); for kfield = 1:numel(fields); extraData.(fields{kfield}) = dlarray(gpuArray( single( this.vectorise_NDto2D(extraData.(fields{kfield}),mask).'))); end
            % fields = fieldnames(pars0);     for kfield = 1:numel(fields); pars0.(fields{kfield}) = dlarray(gpuArray( single( this.vectorise_NDto2D(pars0.(fields{kfield}),mask).'))); end

            % 2.2 display optimisation algorithm parameters
            askadam.display_basic_fitting_parameters(fitting);
            this.display_algorithm_info(fitting)
            
            % 2.3 askAdam optimisation main
            askadamObj = askadam();
            out = askadamObj.optimisation(pars0, @this.modelGradients, data, mask, w, fitting, extraData);

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            disp('The process is completed.')
            
            % clear GPU
            reset(g)
            
        end

        % compute the gradient and loss of forward modelling
        function [gradients,loss,loss_fidelity,loss_reg] = modelGradients(this, parameters, data, mask, weights, fitting, extraData)
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
            % data = this.vectorise_NDto2D(data,mask).';
            
            % Forward model
            [Rreal, Rimag] = this.FWD(askadamObj.rescale_parameters(parameters,fitting.lb,fitting.ub,fitting.model_params),fitting,extraData,mask);
            % [Rreal, Rimag] = this.FWD(parameters,fitting,extraData,mask);
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
            % parameters = askadamObj.scale_parameters(parameters,fitting.lb,fitting.ub,fitting.model_params);
            gradients = dlgradient(loss,parameters);
        
        end
        
        % compute weights for optimisation
        function w = compute_optimisation_weights(this,data,fitting,mask)
        % 
        % Output
        % ------
        % w         : 1D signal masked wegiths that matches the arrangement in masked data later on
        %
            if fitting.isWeighted
                switch lower(fitting.weightMethod)
                    case 'norm'
                       % weights using echo intensity, as suggested in Nam's paper
                        w = sqrt(abs(data));
                    case '1stecho'
                        p = fitting.weightPower;
                        % weights using the 1st echo intensity of each flip angle
                        w = bsxfun(@rdivide,abs(data).^p,abs(data(:,:,:,1)).^p);
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
        function pars0 = estimate_prior(this,data)
        % Estimation starting points 

            % data = zeros(size(data));
            % for kt = 1:length(this.te)
            %     data(:,:,:,kt) = smooth3(data(:,:,:,kt));
            % end

            dims = size(data,1:3);

            % initiate starting point of all parameters
            for k = 1:numel(this.model_params)
                pars0.(this.model_params{k}) = single(this.startpoint(k)*ones(dims));
            end
            
            disp('Estimate starting points based on hybrid fixed points/prior information ...')

            % S0
            [R2s,S0]  = this.R2star_trapezoidal(abs(data),this.te);
            S0(isnan(S0)) = 0; S0(isinf(S0)) = 0; S0(S0<0) = 0;
            pars0.S0 = single(S0);

            % R2*IW
            idx = find(ismember(this.model_params,'R2sIW'));
            R2sIW = R2s - 3;
            R2sIW(isnan(R2sIW)) = single(this.startpoint(idx)); R2sIW(isinf(R2sIW)) = single(this.startpoint(idx)); 
            R2sIW(R2sIW < this.lb(idx)) = single(this.lb(idx)); R2sIW(R2sIW > this.ub(idx)) = single(this.ub(idx));
            pars0.R2sIW = single(R2sIW);

            % R2*EW
            idx = find(ismember(this.model_params,'R2sEW'));
            if ~isempty(idx)
                % if R2*EW is a free parameter then set it
                R2sEW = R2s + 3;
                R2sEW(isnan(R2sEW)) = single(this.startpoint(idx)); R2sEW(isinf(R2sEW)) = single(this.startpoint(idx)); 
                R2sEW(R2sEW < this.lb(idx)) = single(this.lb(idx)); R2sEW(R2sEW > this.ub(idx)) = single(this.ub(idx));
                pars0.R2sEW = single(R2sEW);
            % else
            %     % if R2*EW is not a free parameter then reset R2*EW
            %     idx = find(ismember(this.model_params,'R2sIW'));
            %     pars0.(this.model_params{idx}) = single(this.startpoint(idx)*ones(dims));
            end

            % [~,mwf] = this.superfast_mwi_2m_standard(abs(data),this.te,[]);
            % mwf(mwf>0.1)    = 0.1;     % mainly WM
            % mwf(mwf<0.05)   = 0.02;   % mainly GM
            % mwf(R2s0>35)    = 0.02;
            % mwf             = smooth3(mwf);
            % mwf = single(this.startpoint(2)*ones(dims));
            % mwf(R2s0<15) = 0.05;
            % pars0.(this.model_params{2}) = single(mwf);
            

        end

        %% Signal related functions

        % compute the forward model
        function [Sreal, Simag] = FWD(this, pars, fitting, extraData, mask)
        % Forward model to generate GRE-MWI signal
            if nargin < 5
                S0   = pars.S0;
                mwf  = pars.MWF;
                if fitting.DIMWI.isFitIWF
                    iwf  = pars.IWF;
                else
                    iwf = extraData.IWF;
                end
                r2sMW   = pars.R2sMW;
                r2sIW   = pars.R2sIW;
    
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

                TE = permute(this.te,[2 3 4 1]);
    
            else
                
                % mask out voxels to reduce memory
                S0   = pars.S0(mask).';
                mwf  = pars.MWF(mask).';
                if fitting.DIMWI.isFitIWF
                    iwf  = pars.IWF(mask).';
                else
                    iwf = extraData.IWF;
                end
                r2sMW   = pars.R2sMW(mask).';
                r2sIW   = pars.R2sIW(mask).';

                if fitting.DIMWI.isFitR2sEW
                    r2sEW  = pars.R2sEW(mask).';
                end
                if fitting.DIMWI.isFitFreqMW
                    freqMW  = pars.freqMW(mask).';
                end
                if fitting.DIMWI.isFitFreqIW
                    freqIW  = pars.freqIW(mask).';
                end
                % external effects
                if ~fitting.isComplex % magnitude fitting
                    freqBKG = 0;                          
                    pini    = 0;
                else    % other fittings
                    freqBKG = pars.dfreqBKG(mask).' + extraData.freqBKG; 
                    pini    = pars.dpini(mask).' + extraData.pini;
                end

                extraData.ff    = permute(extraData.ff.',[3 1 2]);
                extraData.theta = permute(extraData.theta.',[3 1 2]);

                TE = gpuArray(dlarray( this.te ));
            end
                
            %%%%%%%%%%%%%%%%%%%% Compartmental Signals %%%%%%%%%%%%%%%%%%%%
            S0MW = S0 .* mwf;
            S0IW = S0 .* (1-mwf) .* iwf;
            S0EW = S0 .* (1-mwf) .* (1-iwf);

            %%%%%%%%%%%%%%%%%%%% DIMWI related operations %%%%%%%%%%%%%%%%%%%%
            
            % if use HCFM to derive either freqMW|freqIW|R2*EW, then computee g-ratio
            if ~fitting.DIMWI.isFitFreqMW || ~fitting.DIMWI.isFitFreqIW || ~fitting.DIMWI.isFitR2sEW
                hcfm_obj = HCFM(this.te,this.B0);

                % g-ratio 
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

            % compute frequency shifts given theta
            if ~fitting.DIMWI.isFitFreqMW 

                % in ppm
                freqMW = hcfm_obj.FrequencyMyelin(this.x_i,this.x_a,g,extraData.theta,this.E) / (this.B0*this.gyro);

            end
            if ~fitting.DIMWI.isFitFreqIW 

                % in ppm
                freqIW = hcfm_obj.FrequencyAxon(this.x_a,g,extraData.theta) / (this.B0*this.gyro);

            end

            freqEW = 0;

            %%%%%%%%%%%%%%%%%%%% Forward model %%%%%%%%%%%%%%%%%%%%
            if nargin < 5
                % Image-based operation
                Sreal = sum((   S0MW .* exp(-TE .* r2sMW) .* cos(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0IW .* exp(-TE .* r2sIW) .* cos(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0EW .* exp(-TE .* r2sEW) .* cos(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini) .* exp(-decayEW) ).*extraData.ff,5);

                Simag = sum((   S0MW .* exp(-TE .* r2sMW) .* sin(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0IW .* exp(-TE .* r2sIW) .* sin(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0EW .* exp(-TE .* r2sEW) .* sin(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini) .* exp(-decayEW) ).*extraData.ff,5);

            else
                % voxel-based operation
                Sreal = sum((   S0MW .* exp(-TE .* r2sMW) .* cos(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0IW .* exp(-TE .* r2sIW) .* cos(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0EW .* exp(-TE .* r2sEW) .* cos(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini) .* exp(-decayEW) ).*extraData.ff,3);
    
                Simag = sum((   S0MW .* exp(-TE .* r2sMW) .* sin(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0IW .* exp(-TE .* r2sIW) .* sin(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0EW .* exp(-TE .* r2sEW) .* sin(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini) .* exp(-decayEW) ).*extraData.ff,3);

                if ~fitting.isComplex
                    Sreal = sqrt(Sreal.^2 + Simag.^2);
                    Simag = 0;
                end
            end

        end
        
        %% Utilities

        % validate extra data
        function [extraData,mask] = validate_data(this,data,extraData,mask,fitting)

           % % check if the signal is monotonic decay
           %  [~,I] = max(abs(data),[],4);
           %  mask = and(mask,I<4);

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
                extraData.pini = angle( data(:,:,:,1) ./ exp(1i* 2*pi*extraData.freqBKG * (this.B0*this.gyro) .* permute(this.te(1),[2 3 4 1])));
            end

            if size(extraData.theta,4) ~= size(extraData.ff,4)
                error('The last dimention of the theta map does not match the last dimension of the fibre fraction map');
            end

            fields = fieldnames(extraData); for kfield = 1:numel(fields); extraData.(fields{kfield}) = single( extraData.(fields{kfield})); end
            
            % thresholding based on single compartment R2*
            [R2s0,~]    = this.R2star_trapezoidal(abs(data),this.te);
            mask        = and(mask,R2s0>this.thres_R2star);

        end

        % normalise input data based on masked signal intensity at 98%
        function [img, scaleFactor] = prepare_data(this,img, mask)

            [~,S0] = this.R2star_trapezoidal(abs(img),this.te);

            scaleFactor = prctile( S0(mask), 98);

            img = img ./ scaleFactor;

        end
        
    end

    methods(Static)
       
        %% signal
        % simple 2-pool matrix inversion
        function [m0,mwf] = superfast_mwi_2m_standard(img,te,t2s)
        %
        % Input
        % --------------
        % img           : multi-echo GRE image, 4D [row,col,slice,TE]
        % te            : echo times in second
        % t2s           : T2* of the two pools, in second, [T2sMW,T2sIEW], if empty
        %                 then default values for 3T will be used
        %
        % Output
        % --------------
        % m0            : proton density of each pool, 4D [row,col,slice,pool]
        % mwf           : myelin water fraction map, range [0,1]
        %
        % Description:  Direct matrix inversion based on simple 2-pool model, i.e.
        %               S(te) = E2s * M0
        %               Useful to estimate initial starting points for MWI fitting
        %
        % Kwok-shing Chan @ DCCN
        % k.chan@donders.ru.nl
        % Date created: 13 Nov 2020
        % Date modified:
        %
        %

            % get size in all image dimensions
            dims = size(img,1:3);
            
            % check input
            if isempty(t2s)
                t2s = [10e-3, 60e-3];   % 3T, [MW, IEW], in second
            end
            
            % T2* decay matrix
            E2s1    = exp(-te(:)/t2s(1));
            E2s2	= exp(-te(:)/t2s(2));
            E2s     = [E2s1,E2s2];
            
            tmp = reshape(abs(img),prod(dims),length(te));
            
            m0 = E2s \ tmp.';
            m0 = reshape(m0.',[dims length(t2s)]);
            
            % compute MWF
            mwf = m0(:,:,:,1) ./ sum(m0,4);
            mwf(mwf<0)      = 0;
            mwf(mwf>1)      = 1;
            mwf(isnan(mwf)) = 0;
            mwf(isinf(mwf)) = 0;
            
            m0(m0 < 0)      = 0;
            m0(isinf(m0))   = 0;
            m0(isnan(m0))   = 0;
        
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

        % vectorise 4D image to 2D with the same last dimention
        function [data, mask_idx] = vectorise_4Dto2D(data,mask)

            dims = size(data,[1 2 3]);

            if nargin < 2
                mask = ones(dims);
            end

             % vectorise data
            data        = reshape(data,prod(dims),size(data,4));
            mask_idx    = find(mask>0);
            data        = data(mask_idx,:);

            if ~isreal(data)
                data = cat(2,real(data),imag(data));
            end

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