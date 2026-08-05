classdef gpuMCRMWI < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 22 September 2024
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
        model_params    = { 'S0';   'MWF';  'IWF'; 'R1IEW'; 'kIEWM'; 'R2sMW'; 'R2sIW'; 'R2sEW'; 'freqMW';'freqIW';'dfreqBKG';'dpini'};
        ub              = [    2;     0.3;      1;       2;      10;     200;      50;      50;     0.25;    0.05;       0.4;   pi/2];
        lb              = [ 1e-8;    1e-8;   1e-8;     1/3;    1e-8;      50;1/150e-3;1/150e-3;    -0.05;    -0.1;      -0.4;  -pi/2];
        startpoint      = [    1;     0.1;    0.8;       1;    1e-8;     100;      15;      21;     0.04;    1e-8;      1e-8;   1e-8];

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

        thres_R2star    = 150;  % 1/s, upper bound
        thres_T1        = 3.1;    % s, upper bound

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
                if isfield(fixed_params,'x_i');         this.x_i            = single(fixed_params.x_i);             end
                if isfield(fixed_params,'x_a');         this.x_a            = single(fixed_params.x_a);             end
                if isfield(fixed_params,'E');           this.E              = single(fixed_params.E);               end
                if isfield(fixed_params,'rho_mw');      this.rho_mw         = single(fixed_params.rho_mw);          end
                if isfield(fixed_params,'B0');          this.B0             = single(fixed_params.B0);              end
                if isfield(fixed_params,'B0dir');       this.B0dir          = single(fixed_params.B0dir);           end
                if isfield(fixed_params,'t1_mw');       this.t1_mw          = single(fixed_params.t1_mw);           end
                if isfield(fixed_params,'thres_R2s');   this.thres_R2star   = single(fixed_params.thres_R2star);    end
                if isfield(fixed_params,'thres_T1');    this.thres_T1       = single(fixed_params.thres_T1);        end
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
            
            disp('----------------------')
            disp('Parameters to be fixed')
            disp('----------------------')
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
            fitting = this.check_set_default(fitting,data);

            % get matrix size
            dims = size(data,1:3);

            % make sure input data are valid
            [extraData,mask] = this.validate_data(data,extraData,mask,fitting);

            % normalised data if needed
            [data, scaleFactor] = this.prepare_data(data,mask, extraData.b1);

            % mask sure no nan or inf
            [data,mask] = utils.remove_img_naninf(data,mask);

            % convert datatype to single
            data    = single(data);
            mask    = mask > 0;

            % determine if we need to divide the data to fit in GPU: TODO
            g = gpuDevice; reset(g);
            memoryFixPerVoxel       = 0.0001;   % get this number based on mdl fit
            memoryDynamicPerVoxel   = 0.0001;     % get this number based on mdl fit
            [NSegment,maxSlice]     = utils.find_optimal_divide(mask,memoryFixPerVoxel,memoryDynamicPerVoxel);

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
                fields      = fieldnames(extraData); for kfield = 1:numel(fields); extraData_tmp.(fields{kfield}) = extraData.(fields{kfield})(:,:,slice,:,:); end

                % run fitting
                [out_tmp]    = this.fit(dwi_tmp,mask_tmp,fitting,extraData_tmp);

                % restore 'out' structure from segment
                out = utils.restore_segment_structure(out,out_tmp,slice,ks);

            end
            out.mask = mask;
            % rescale S0
            out.final.S0    = out.final.S0 *scaleFactor;
            out.min.S0      = out.min.S0 *scaleFactor;

            % save the estimation results if the output filename is provided
            askadam.save_askadam_output(fitting.outputFilename,out)

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
            
            % check GPU
            gpool = gpuDevice;
            
            % get image size
            dims = size(data,1:3);

            %%%%%%%%%%%%%%%%%%%% 1. Validate and parse input %%%%%%%%%%%%%%%%%%%%
            if nargin < 3 || isempty(mask); mask = ones(dims,'logical'); end % if no mask input then fit everthing
            if nargin < 4; fitting = struct(); end

            % get all fitting algorithm parameters 
            fitting                 = this.check_set_default(fitting,data);
            % determine fitting parameters
            this                    = this.updateProperty(fitting);
            fitting.model_params    = this.model_params;
            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(this.model_params)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(this.model_params)); end
            
            % set initial starting points
            pars0 = this.estimate_prior(data,mask,extraData);

            % load pre-trained ANN
            ann_epgx_phase = load(fitting.epgx_phase_ann,'dlnet');  ann_epgx_phase.dlnet.alpha  = 0.01;
            ann_epgx_magn  = load(fitting.epgx_mag_ann,'dlnet');    ann_epgx_magn.dlnet.alpha   = 0.01;
            
            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w = this.compute_optimisation_weights(data,fitting); % This is a customised funtion

            % split data into real and imaginary parts for complex-valued data
            if fitting.isComplex; data = cat(6,real(data),imag(data)); end

            % 2.2 display optimisation algorithm parameters
            this.display_algorithm_info(fitting)

            % 3. askAdam optimisation main
            askadamObj  = askadam();

            % 3.1. initial global optimisation
            disp('##############################################')
            disp('1. Runnning optimisation on all voxels...')

            % askadam.optimisation does not see extractData so we have to manually mask the data inside here, make sure the voxel is on the second dimension
            extraData_fit   = structfun(@transpose, utils.gpu_vectorise_NDto2D_struct(extraData,mask) ,'UniformOutput',false);
            
            % get residual of the initial starting point
            fitting0    = fitting; fitting0.iteration = 0; fitting0.isdisplay = false;
            out0        = askadamObj.optimisation(data, mask, w, pars0, fitting0, @this.FWD, fitting, extraData_fit, ann_epgx_phase.dlnet, ann_epgx_magn.dlnet);

            % run optimisation
            out_init    = askadamObj.optimisation(data, mask, w, pars0, fitting, @this.FWD, fitting, extraData_fit, ann_epgx_phase.dlnet, ann_epgx_magn.dlnet);

            % find bad fit voxels
            mask_badfit = out_init.final.resloss >= out0.final.resloss; % utils.reshape_ND2image(mean(abs(utils.vectorise_NDto2D(data.*w,ones(size(mask)))),2),ones(size(mask))); 

            disp('Optimisation completed.');
            disp('##############################################')

            % whether to further refine the fitting
            if fitting.isMultiStep
                mask_outlier                = isoutlier(out_init.final.resloss);
                mask_outlier(mask_badfit)   = 1;

                % inliners
                disp('##############################################')
                disp('2. Further optimising non-outlier voxels...')

                
                mask_fit    = and(mask, mask_outlier==0 ); % update fitting mask
                pars0_fit   = []; fields = fieldnames(pars0); for kf = 1:numel(fields); pars0_fit.(fields{kf}) = out_init.final.(fields{kf}); end % update starting positions
                
                % refine inliner
                extraData_fit   = structfun(@transpose, utils.gpu_vectorise_NDto2D_struct(extraData,mask_fit) ,'UniformOutput',false);

                reductionFactor = 1; attempt = 1;
                while attempt <=2 
                    % update learning rate if needed (to avoid being too large)
                    fitting_refine = fitting; fitting_refine.initialLearnRate = fitting.initialLearnRate / reductionFactor; if fitting_refine.initialLearnRate ~= fitting.initialLearnRate; disp('Leaarning rate is updated'); end
                    out_inliners    = askadamObj.optimisation(data, mask_fit, w, pars0_fit, fitting_refine, @this.FWD, fitting, extraData_fit, ann_epgx_phase.dlnet, ann_epgx_magn.dlnet);
                    if out_inliners.final.Niteration < min(100, fitting_refine.iteration)
                        attempt         = attempt + 1;
                        reductionFactor = reductionFactor*2;    % if too fewer iterations then reduces learning rate
                    else
                        break
                    end
                end
                
                % find bad fit voxels
                mask_badfit_inliners = out_inliners.final.resloss - out_init.final.resloss > 1e-4;

                disp('Optimisation completed.');
                disp('##############################################')
                
                % outliers
                disp('##############################################')
                disp('3. Re-fitting outliers voxels...')

                mask_fit = and(mask,  mask_outlier>0 ); % update fitting mask
                
                extraData_fit = structfun(@transpose, utils.gpu_vectorise_NDto2D_struct(extraData,mask_fit) ,'UniformOutput',false);
                out_outliners = askadamObj.optimisation(data, mask_fit, w, pars0, fitting, @this.FWD, fitting, extraData_fit, ann_epgx_phase.dlnet, ann_epgx_magn.dlnet);

                % find bad fit voxels
                mask_badfit_outliners =  out_outliners.final.resloss - out_init.final.resloss > 1e-4;

                disp('Optimisation completed.');
                disp('##############################################')

                % Combine and run a final optimisation
                disp('##############################################')
                disp('4. Combining final result...')
                pars0_fit = []; fields = fieldnames(pars0); for kf = 1:numel(fields); pars0_fit.(fields{kf}) = out_init.final.(fields{kf}) .* and(mask,or(mask_badfit_inliners,mask_badfit_outliners)) + ...
                                                                                                                        out_inliners.final.(fields{kf}).*and(mask,~mask_outlier.*~mask_badfit_inliners ) + ...
                                                                                                                        out_outliners.final.(fields{kf}).*and(mask,mask_outlier.*~mask_badfit_outliners); end
                               
                fitting_final   = fitting; fitting_final.iteration = 0; fitting_final.isdisplay = 0;
                extraData_fit   = structfun(@transpose, utils.gpu_vectorise_NDto2D_struct(extraData,mask) ,'UniformOutput',false);
                out             = askadamObj.optimisation(data, mask, w, pars0_fit, fitting_final, @this.FWD, fitting, extraData_fit, ann_epgx_phase.dlnet, ann_epgx_magn.dlnet);

                out.step1 = out_init.final;
                
            else
                out = out_init;
            end

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            disp('The process is completed.')
            disp('##############################################')
            
            % clear GPU
            reset(gpool)
            
        end

        % compute weights for optimisation
        function w = compute_optimisation_weights(this,data,fitting)
        % 
        % Output
        % ------
        % w         : ND signal masked wegiths
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

            % separate real/imaginary parts into 6th dim
            if fitting.isComplex
                w = repmat(w,1,1,1,1,1,2);
            end
        end

        %% Prior estimation related functions

        % estimate starting points
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
            pars0.dfreqBKG = repmat(pars0.dfreqBKG,[1,1,1,size(extraData.freqBKG,4)]);

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

            % [~,mwf] = this.superfast_mwi_2m_mcr(abs(data),[],mask,extraData.b1,1);
            % % mwf = mwf + 0.02;
            % mwf = min(mwf,0.25);
            % mwf = max(mwf,0.05);
            % % mwf = max(mwf,this.lb(2));
            % % mwf(mwf>0.15)                   = 0.15;
            % % mwf(and(mwf>=0.015,mwf<=0.1))   = 0.1;
            % % mwf(mwf < 0.05)                 = 0.05;
            % pars0.(this.model_params{2})    = single(mwf);

        end

        function [m0,mwf] = superfast_mwi_2m_mcr(this,img,t2s,mask,b1map, isSuperfast)
        % [m0,mwf,t2s_iew,t1_iew] = superfast_mwi_2m_mcr_self(img,te,fa,tr,t2s,t1_mw,mask,b1map,mode)
        %
        % Input
        % --------------
        % img           : variable flip angle multi-echo GRE image, 5D [row,col,slice,TE,Flip angle]
        % te            : echo times in second
        % fa            : flip angle in degree
        % tr            : repetition time in second
        % t2s           : T2* of the two pools, in second, [T2sMW,T2sIEW], if empty
        %                 then default values for 3T will be used
        % t1_mw         : T1 of MW, in second, if empty
        %                 then a default value for 3T will be used
        % mask          : signal mask, (optional)
        % b1map         : B1 flip angel ratio map, (optional)
        % mode          : IEW T1 estimation approach, ('superfast' or 'normal')
        %
        % Output
        % --------------
        % m0            : proton density of each pool, 4D [row,col,slice,pool]
        % mwf           : myelin water fraction map, range [0,1]
        %
        % Description:  Direct matrix inversion based on simple 2-pool model, i.e.
        %               S(te,fa) = E1 * M0 * E2s
        %               Useful to estimate initial starting points for MWI fitting
        %
        % Kwok-shing Chan @ DCCN
        % k.chan@donders.ru.nl
        % Date created: 12 Nov 2020
        % Date modified:
        %
        %
        % get size in all image dimensions
        dims = size(img,1:3);
        
        % check input
        if isempty(t2s)
            t2s = [10e-3, 60e-3];   % 3T, [MW, IEW], in second
        end
        if nargin < 5 || isempty(b1map)
            b1map = ones(dims);
        end
        if nargin < 4 || isempty(mask)
            mask = ones(dims);
        end

        % % IEW T1 estimation is relative insensitive to the change of its T2*
        despot1_obj         = despot1(this.tr,this.fa);
        % [t1_iew, ~, ~]      = despot1_obj.estimate(permute(abs(img(:,:,:,1,:)),[1 2 3 5 4]), mask, b1map);
        % t1_iew(t1_iew>5)    = 5;
        % mask_nonCSF         = and(t1_iew>0.1,t1_iew<5);
        % thres               = median(t1_iew(mask_nonCSF),"omitmissing") + iqr(t1_iew(mask_nonCSF));
        % t1_iew(t1_iew<thres) = thres;
        
        % assign T2* to IEW
        ind     = find(this.te>1.5*t2s(1));
        if isempty(ind)
            t2s_iew = zeros([dims length(this.fa)]);
            for kk = 1:length(this.fa)
                [~,~,t2s_iew(:,:,:,kk)] = this.R2star_trapezoidal(abs(img(:,:,:,:,kk)),this.te);
            end
        else
            t2s_iew = zeros([dims length(this.te)-length(ind) length(this.fa)]);
            for kk = 1:length(this.fa)
                for k = 1:length(this.te)-length(ind)
                    [~,~,t2s_iew(:,:,:,k,kk)] = this.R2star_trapezoidal(abs(img(:,:,:,k:end,kk)),this.te);
                end
            end
        end
        t2s_iew = reshape(t2s_iew,[dims size(t2s_iew,4)*size(t2s_iew,5)]);
        t2s_iew = median(t2s_iew,4); % t2s_iew = max(t2s_iew,[],4);

        t2s(2)  = median(t2s_iew(t2s_iew<200e-3),"omitmissing") + iqr(t2s_iew(t2s_iew<200e-3));

        if isSuperfast
            % T2* decay matrix
            E2s1    = exp(-this.te(:)/t2s(1));
            E2s2	= exp(-this.te(:)/t2s(2));
            E2s     = [E2s1,E2s2];
    
            s0_iew  = zeros([dims length(this.fa)]);
            for kfa = 1:length(this.fa)
    
                tmp = reshape(abs(img(:,:,:,:,kfa)),prod(dims),length(this.te));
    
                s0 = E2s \ tmp.';
    
                s0_iew(:,:,:,kfa)   = reshape(s0(2,:),dims);
            end
            
        else
            % get IEW signal amplitude for DESPOT1, which should be more robust
            % s0_mw   = zeros([dims length(fa)]);
            s0_iew  = zeros([dims length(this.fa)]);
            for kfa = 1:length(this.fa)
    
            tmp = reshape(abs(img(:,:,:,:,kfa)),prod(dims),length(this.te));
    
            s0 = zeros(2,prod(dims));
            for k = 1:prod(dims)
                if mask(k) ~= 0
                    % T2* decay matrix
                    E2s = [exp(-this.te(:)/t2s(1)),exp(-this.te(:)/t2s_iew(k))];
        
                    s0(:,k) = E2s \ tmp(k,:).';
                end
            end
            % s0_mw(:,:,:,kfa)    = reshape(s0(1,:),dims);
            s0_iew(:,:,:,kfa)   = reshape(s0(2,:),dims);
            end
        end
        % IEW T1 estimation is relative insensitive to the change of its T2*
        [t1_iew, ~, ~] = despot1_obj.estimate(s0_iew, mask, b1map);
        
        % main
        tmp     = reshape(abs(img),prod(dims),length(this.te),length(this.fa));
        m0      = zeros([prod(dims),2]);
        for k = 1:prod(dims)
            if mask(k) ~= 0
            
                % T1-T2* signal
                temp = squeeze(tmp(k,:,:)).';
        
                % T2* decay matrix
                if isSuperfast
                    E2s = [exp(-this.te(:).'/t2s(1));exp(-this.te(:).'/t2s(2))];
                else
                    E2s = [exp(-this.te(:).'/t2s(1));exp(-this.te(:).'/t2s_iew(k))];
                end
                
                % T1 steady-state matrix
                E1_mw  = sind(this.fa(:)*b1map(k)) .* (1-exp(-this.tr/this.t1_mw))./(1-cosd(this.fa(:)*b1map(k))*exp(-this.tr/this.t1_mw));
                E1_iew = sind(this.fa(:)*b1map(k)) .* (1-exp(-this.tr/t1_iew(k)))./(1-cosd(this.fa(:)*b1map(k))*exp(-this.tr/t1_iew(k)));
                E1 = [E1_mw,E1_iew];
                
                % matrix inversion
        %         temp2 = pinv(E1)*temp*pinv(E2s);
                temp2 = (E1 \ temp) / (E2s);
                
                % the diagonal components represent signal amplitude
                m0(k,:) =  diag(temp2);
            end
        end
        
        
        m0 = reshape(m0,[dims 2]);
        
        % compute MWF
        mwf = m0(:,:,:,1) ./ sum(m0,4);
        mwf(mwf<0)      = 0; mwf(isnan(mwf)) = 0; mwf(isinf(mwf)) = 0;
        m0(m0 < 0)      = 0; m0(isinf(m0))   = 0; m0(isnan(m0))   = 0;
        
        end
        
        %% Signal related functions

        % compute the forward model
        function [s] = FWD(this, pars, mask, fitting, extraData, dlnet_phase,dlnet_magn)

            nFA = numel(this.fa);

        % Forward model to generate NEXI signal
            if isempty(mask)
                S0   = pars.S0;
                mwf  = pars.MWF;
                if fitting.DIMWI.isFitIWF; iwf  = pars.IWF; else; iwf = extraData.IWF; end
                r2sMW   = pars.R2sMW;
                r2sIW   = pars.R2sIW;
                r1iew   = pars.R1IEW;

                if fitting.isFitExchange;       kiewm = pars.kIEWM; end
    
                if fitting.DIMWI.isFitR2sEW;    r2sEW  = pars.R2sEW;    end
                if fitting.DIMWI.isFitFreqMW;   freqMW  = pars.freqMW;  end
                if fitting.DIMWI.isFitFreqIW;   freqIW  = pars.freqIW;  end
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
                S0   = utils.row_vector(pars.S0(mask));
                mwf  = utils.row_vector(pars.MWF(mask));
                if fitting.DIMWI.isFitIWF;  iwf  = utils.row_vector(pars.IWF(mask)); else; iwf = extraData.IWF; end
                r2sMW   = utils.row_vector(pars.R2sMW(mask));
                r2sIW   = utils.row_vector(pars.R2sIW(mask));
                r1iew   = utils.row_vector(pars.R1IEW(mask));

                if fitting.isFitExchange;       kiewm   = utils.row_vector(pars.kIEWM(mask));     end
    
                if fitting.DIMWI.isFitR2sEW;    r2sEW   = utils.row_vector(pars.R2sEW(mask));     end
                if fitting.DIMWI.isFitFreqMW;   freqMW  = utils.row_vector(pars.freqMW(mask));    end
                if fitting.DIMWI.isFitFreqIW;   freqIW  = utils.row_vector(pars.freqIW(mask));    end
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
                    freqBKG = shiftdim((utils.vectorise_NDto2D(pars.dfreqBKG,mask).' + extraData.freqBKG).',dshift) ; 
                    pini    = utils.row_vector(pars.dpini(mask)) + extraData.pini;
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
            totalVolume = S0IW + S0EW + S0MW/this.rho_mw; totalVolume = max(totalVolume,askadam.epsilon); % avoid division by zeros
            MVF         = (S0MW/this.rho_mw) ./ totalVolume; MVF = max(MVF,askadam.epsilon); % avoid division by zeros in BM calculation

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
                if ismatrix(pars.MWF)
                    S0MW  = permute(S0MW,[1 2 5 4 3]);
                    S0IEW = permute(S0IEW,[1 2 5 4 3]);
                    % S0IW = permute(S0IW,[1 2 5 4 3]);
                    % S0EW = permute(S0EW,[1 2 5 4 3]);
                    freqBKG = permute(freqBKG,[3 2 4 5 1]);
                    S0IEW_phase = permute(S0IEW_phase,[1 2 5 4 3]);
                end
                Sreal = sum((   S0MW            .* exp(-TE .* r2sMW) .* cos(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...               % MW
                                S0IEW.*iwf      .* exp(-TE .* r2sIW) .* cos(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini + S0IEW_phase) + ...    % IW
                                S0IEW.*(1-iwf)  .* exp(-TE .* r2sEW) .* cos(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini + S0IEW_phase) .* exp(-decayEW) ).*extraData.ff,6);
    
                Simag = sum((   S0MW            .* exp(-TE .* r2sMW) .* sin(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0IEW.*iwf      .* exp(-TE .* r2sIW) .* sin(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini + S0IEW_phase) + ...
                                S0IEW.*(1-iwf)  .* exp(-TE .* r2sEW) .* sin(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini + S0IEW_phase) .* exp(-decayEW) ).*extraData.ff,6);

                % Sreal = sum((   S0MW .* exp(-TE .* r2sMW) .* cos(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...
                %                 S0IW .* exp(-TE .* r2sIW) .* cos(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini) + ...
                %                 S0EW .* exp(-TE .* r2sEW) .* cos(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini) .* exp(-decayEW) ).*extraData.ff,6);
                % 
                % Simag = sum((   S0MW .* exp(-TE .* r2sMW) .* sin(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...
                %                 S0IW .* exp(-TE .* r2sIW) .* sin(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini) + ...
                %                 S0EW .* exp(-TE .* r2sEW) .* sin(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini) .* exp(-decayEW) ).*extraData.ff,6);

                s = cat(6,Sreal,Simag);

            else
                % voxel-based operation
                Sreal = sum((   S0MW            .* exp(-TE .* r2sMW) .* cos(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...               % MW
                                S0IEW.*iwf      .* exp(-TE .* r2sIW) .* cos(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini + S0IEW_phase) + ...    % IW
                                S0IEW.*(1-iwf)  .* exp(-TE .* r2sEW) .* cos(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini + S0IEW_phase) .* exp(-decayEW) ).*extraData.ff,4);
    
                Simag = sum((   S0MW            .* exp(-TE .* r2sMW) .* sin(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...
                                S0IEW.*iwf      .* exp(-TE .* r2sIW) .* sin(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini + S0IEW_phase) + ...
                                S0IEW.*(1-iwf)  .* exp(-TE .* r2sEW) .* sin(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini + S0IEW_phase) .* exp(-decayEW) ).*extraData.ff,4);

                if ~fitting.isComplex
                    s = permute( sqrt(Sreal.^2 + Simag.^2), [1 3 2]);
                    
                else
                    % Sreal = reshape( permute( Sreal, [2 1 3]), numel(MVF), numel(TE)*numel(FA)).';
                    % Simag = reshape( permute( Simag, [2 1 3]), numel(MVF), numel(TE)*numel(FA)).';
                    
                    s = cat(1,reshape( permute( Sreal, [2 1 3]), numel(MVF), numel(TE)*numel(FA)).',...     % real
                              reshape( permute( Simag, [2 1 3]), numel(MVF), numel(TE)*numel(FA)).');       % imaginary
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

            fields = fieldnames(extraData); for kfield = 1:numel(fields); extraData.(fields{kfield}) = single( extraData.(fields{kfield})); end

            despot1_obj          = despot1(this.tr,this.fa);
            [t1, ~, mask_fitted] = despot1_obj.estimate(permute(abs(data(:,:,:,1,:)),[1 2 3 5 4]), mask, extraData.b1);
            [R2star,~]           = R2star_trapezoidal(mean(abs(data),5),this.te);

            % DIMWI
            if ~fitting.DIMWI.isFitFreqMW || ~fitting.DIMWI.isFitFreqIW || ~fitting.DIMWI.isFitR2sEW
                % fibre fraction
                if isfield(extraData,'ff')
                    extraData.ff                        = bsxfun(@rdivide,extraData.ff,sum(extraData.ff,4));
                    mask                                = and(mask,min(~isnan(extraData.ff),[],4));
                    extraData.ff(isnan(extraData.ff))   = 0;
                else
                    error('Fibre fraction map is required for MCR-DIMWI!');
                end
                % fibre orientation
                if ~isfield(extraData,'theta')
                    if ~isfield(extraData,'fo')
                        error('Fibre orientation map is required for MCR-DIMWI!');
                    else
                        fo    = double(extraData.fo); % fibre orientation w.r.t. B0
                        theta = zeros(size(extraData.ff));
                        for kfo = 1:size(fo,5)
                            theta(:,:,:,kfo) = this.AngleBetweenV1MapAndB0(fo(:,:,:,:,kfo),this.B0dir);
                        end
                        extraData.theta = single(theta);
                        extraData = rmfield(extraData,"fo");
                    end
                end
            else
                extraData.theta = zeros(dims,'single');
                extraData.ff    = ones(dims,'single');
            end
            if size(extraData.theta,4) ~= size(extraData.ff,4)
                error('The last dimention of the theta map does not match the last dimension of the fibre fraction map');
            end

            % final mask
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
        function [S0M, S0IEW] = model_BM_2T1_matlab(TR,true_famp,f,R1f,R1r,kfr)
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
        function [R2star,S0,T2star] = R2star_trapezoidal(img,te)
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
            T2star = temp./(img(:,:,:,1)-img(:,:,:,end));
                
            R2star = 1./T2star;

            S0 = img(1:(numel(img)/dims(end)))'.*exp(R2star(:)*te(1));
            if numel(S0) ~=1
                S0 = reshape(S0,dims(1:end-1));
            end
        end

        %% Utilities
        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default(fitting,data)
            % get basic fitting setting check
            fitting2 = askadam.check_set_default_basic(fitting);

            % check weighted sum of cost function
            if ~isfield(fitting,'isWeighted');      fitting2.isWeighted     = true; end
            if ~isfield(fitting,'weightMethod');    fitting2.weightMethod   = '1stecho'; end
            if ~isfield(fitting,'weightPower');     fitting2.weightPower    = 2; end
            
            % check hollow cylinder fibre model parameters
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitFreqMW');  fitting2.DIMWI.isFitFreqMW  = true; end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitFreqIW');  fitting2.DIMWI.isFitFreqIW  = true; end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitR2sEW');   fitting2.DIMWI.isFitR2sEW   = true; end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitIWF');     fitting2.DIMWI.isFitIWF     = true; end

            if ~isfield(fitting,'isFitExchange');   fitting2.isFitExchange  = true; end
            if ~isfield(fitting,'isEPG');           fitting2.isEPG          = true; end

            % get customised fitting setting check
            if ~isfield(fitting,'regmap');      fitting2.regmap = 'MWF'; end

            if ~isfield(fitting,'isComplex');   fitting2.isComplex = true; end

            if isreal(data);    fitting2.isComplex = false;  end

            % get ANN paths
            scriptPath = fileparts(mfilename('fullpath'));
            if ~isfield(fitting,'epgx_phase_ann');   fitting2.epgx_phase_ann  = fullfile(scriptPath,'EPGXgen_net','MCRMWI_MLP_EPGX_RFphase50_T1M234_phase.mat');  end
            if ~isfield(fitting,'epgx_mag_ann');     fitting2.epgx_mag_ann    = fullfile(scriptPath,'EPGXgen_net','MCRMWI_MLP_EPGX_RFphase50_T1M234_magn.mat');  end

            if ~isfield(fitting,'isMultiStep');     fitting2.isMultiStep    = false;  end

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