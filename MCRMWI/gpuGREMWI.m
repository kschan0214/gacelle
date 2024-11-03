classdef gpuGREMWI < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 22 July 2024
% Date modified: 22 September 2024

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
        modelParams     = { 'S0';   'MWF';  'IWF';  'R2sMW';'R2sIW';'R2sEW'; 'freqMW';'freqIW';'dfreqBKG';'dpini'};
        ub              = [    2;     0.3;      1;      200;     50;     50;     0.25;    0.05;       0.4;   pi/2];
        lb              = [ 1e-8;    1e-8;   1e-8;       50;      2;      2;    -0.05;    -0.1;      -0.4;  -pi/2];
        startPoint      = [    1;     0.1;    0.6;      100;     15;     21;     0.04;       0;         0;      0];

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
                if isfield(fixed_params,'x_i');         this.x_i            = single(fixed_params.x_i);             end
                if isfield(fixed_params,'x_a');         this.x_a            = single(fixed_params.x_a);             end
                if isfield(fixed_params,'E');           this.E              = single(fixed_params.E);               end
                if isfield(fixed_params,'rho_mw');      this.rho_mw         = single(fixed_params.rho_mw);          end
                if isfield(fixed_params,'B0');          this.B0             = single(fixed_params.B0);              end
                if isfield(fixed_params,'B0dir');       this.B0dir          = single(fixed_params.B0dir);           end
                if isfield(fixed_params,'thres_R2s');   this.thres_R2star   = single(fixed_params.thres_R2star);    end
            end
        end
        
        % update properties according to lmax
        function this = updateProperty(this, fitting)

            if fitting.isComplex == 0
                for kpar = {'dfreqBKG','dpini'}
                    idx = find(ismember(this.modelParams,kpar));
                    this.modelParams(idx)    = [];
                    this.lb(idx)              = [];
                    this.ub(idx)              = [];
                    this.startPoint(idx)      = [];
                end
            end

            % DIMWI
            if fitting.DIMWI.isFitFreqIW == 0
                idx = find(ismember(this.modelParams,'freqIW'));
                this.modelParams(idx)    = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startPoint(idx)      = [];
            end

            if fitting.DIMWI.isFitFreqMW == 0
                idx = find(ismember(this.modelParams,'freqMW'));
                this.modelParams(idx)    = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startPoint(idx)      = [];
            end

            if fitting.DIMWI.isFitIWF == 0
                idx = find(ismember(this.modelParams,'IWF'));
                this.modelParams(idx)    = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startPoint(idx)      = [];
            end

            if fitting.DIMWI.isFitR2sEW == 0
                idx = find(ismember(this.modelParams,'R2sEW'));
                this.modelParams(idx)    = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startPoint(idx)      = [];
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
            disp('---------------------')

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
            fitting = this.check_set_default(fitting,data);

            % get matrix size
            dims = size(data,1:3);

            % make sure input data are valid
            [extraData,mask] = this.validate_data(data,extraData,mask,fitting);

            % compute rotationally invariant signal if needed
            [data, scaleFactor] = this.prepare_data(data,mask);

            % mask sure no nan or inf
            [data,mask] = utils.remove_img_naninf(data,mask);

            % convert datatype to single
            data    = single(data);
            mask    = mask >0;

            % determine if we need to divide the data to fit in GPU
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
                dwi_tmp     = data(:,:,slice,:);
                mask_tmp    = mask(:,:,slice);
                fields      = fieldnames(extraData); 
                for kfield = 1:numel(fields); extraData_tmp.(fields{kfield}) = single(extraData.(fields{kfield})(:,:,slice,:,:)); end

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
            fitting             = this.check_set_default(fitting,data);
            % determine fitting parameters
            this                = this.updateProperty(fitting);
            fitting.modelParams = this.modelParams;
            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(this.modelParams)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(this.modelParams)); end
            
            % set initial starting points
            pars0 = this.determine_x0(data,fitting);
            % pars0 = this.estimate_prior(data);
            
            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w = this.compute_optimisation_weights(data,fitting); % This is a customised funtion

            % split data into real and imaginary parts for complex-valued data
            if fitting.isComplex; data = cat(5,real(data),imag(data)); end

            % 2.2 display optimisation algorithm parameters
            this.display_algorithm_info(fitting)

            % 3. askAdam optimisation main
            askadamObj  = askadam();
            % % mask out data to reduce memory load
            % data = utils.vectorise_NDto2D(data,mask).';
            % if ~isempty(w); w = utils.vectorise_NDto2D(w,mask).'; end
            % fieldname = fieldnames(extraData); for km = 1:numel(fieldname); extraData.(fieldname{km}) = gpuArray(single( utils.vectorise_NDto2D(extraData.(fieldname{km}),mask) ).'); end
            extraData   = utils.masking_ND2AD_preserve_struct(extraData,mask) ;
            out         = askadamObj.optimisation(data, mask, w, pars0, fitting, @this.FWD, fitting, extraData);

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            disp('The process is completed.')
            
            % clear GPU
            reset(gpool)
            
        end

        %% Prior estimation related functions

        % determine how the starting points will be set up
        function x0 = determine_x0(this,y,fitting) 

            disp('---------------');
            disp('Starting points');
            disp('---------------');

            dims = size(y,1:3);

            if ischar(fitting.start)
                switch lower(fitting.start)
                    case 'prior'
                        % using maximum likelihood method to estimate starting points
                        x0 = this.estimate_prior(y);
    
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
        function pars0 = estimate_prior(this,data)
        % Estimation starting points 

            % data = zeros(size(data));
            % for kt = 1:length(this.te)
            %     data(:,:,:,kt) = smooth3(data(:,:,:,kt));
            % end

            dims = size(data,1:3);

            % initiate starting point of all parameters
            pars0 = utils.initialise_x0(dims,this.modelParams,this.startPoint);
            
            disp('Estimate starting points based on hybrid fixed points/prior information ...')

            % S0
            [R2s,S0]  = this.R2star_trapezoidal(abs(data),this.te);
            S0(isnan(S0)) = 0; S0(isinf(S0)) = 0; S0(S0<0) = 0;
            pars0.S0 = single(S0);

            % R2*IW
            idx = find(ismember(this.modelParams,'R2sIW'));
            R2sIW = R2s - 3;
            R2sIW(isnan(R2sIW)) = single(this.startPoint(idx)); R2sIW(isinf(R2sIW)) = single(this.startPoint(idx)); 
            R2sIW(R2sIW < this.lb(idx)) = single(this.lb(idx)); R2sIW(R2sIW > this.ub(idx)) = single(this.ub(idx));
            pars0.R2sIW = single(R2sIW);

            % R2*EW
            idx = find(ismember(this.modelParams,'R2sEW'));
            if ~isempty(idx)
                % if R2*EW is a free parameter then set it
                R2sEW = R2s + 3;
                R2sEW(isnan(R2sEW)) = single(this.startPoint(idx)); R2sEW(isinf(R2sEW)) = single(this.startPoint(idx)); 
                R2sEW(R2sEW < this.lb(idx)) = single(this.lb(idx)); R2sEW(R2sEW > this.ub(idx)) = single(this.ub(idx));
                pars0.R2sEW = single(R2sEW);
            % else
            %     % if R2*EW is not a free parameter then reset R2*EW
            %     idx = find(ismember(this.modelParams,'R2sIW'));
            %     pars0.(this.modelParams{idx}) = single(this.startpoint(idx)*ones(dims));
            end

            % [~,mwf] = this.superfast_mwi_2m_standard(abs(data),this.te,[]);
            % mwf(mwf>0.15)                   = 0.15;         
            % mwf(and(mwf>=0.05,mwf<=0.1))    = 0.1;   
            % mwf(mwf<0.015)                  = 0.03;
            % pars0.(this.modelParams{2})    = single(mwf);
            

        end

        %% Signal related functions

        % Forward model to generate GRE-MWI signal
        function [s] = FWD(this, pars, fitting, extraData)

            TE = gpuArray(dlarray( permute(this.te, [2 3 4 1] )));              % TE always on 4th dim

            S0   = pars.S0;
            mwf  = pars.MWF;
            if fitting.DIMWI.isFitIWF; iwf  = pars.IWF; else; iwf = extraData.IWF; end
            r2sMW   = pars.R2sMW;
            r2sIW   = pars.R2sIW;

            if fitting.DIMWI.isFitR2sEW;    r2sEW   = pars.R2sEW;   end
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

            Sreal = sum((   S0MW .* exp(-TE .* r2sMW) .* cos(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...
                            S0IW .* exp(-TE .* r2sIW) .* cos(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini) + ...
                            S0EW .* exp(-TE .* r2sEW) .* cos(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini) .* exp(-decayEW) ).*extraData.ff,5);

            Simag = sum((   S0MW .* exp(-TE .* r2sMW) .* sin(TE .* 2.*pi.*(freqMW+freqBKG).*this.B0.*this.gyro + pini) + ...
                            S0IW .* exp(-TE .* r2sIW) .* sin(TE .* 2.*pi.*(freqIW+freqBKG).*this.B0.*this.gyro + pini) + ...
                            S0EW .* exp(-TE .* r2sEW) .* sin(TE .* 2.*pi.*(freqEW+freqBKG).*this.B0.*this.gyro + pini) .* exp(-decayEW) ).*extraData.ff,5);

            if ~fitting.isComplex
                s = sqrt(Sreal.^2 + Simag.^2);
            else
                s = cat(5,Sreal,Simag);
            end

            % vectorise to match maksed measurement data
            s = utils.reshape_ND2AD(s,[]);

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

            fields = fieldnames(extraData); for kfield = 1:numel(fields); extraData.(fields{kfield}) = single( extraData.(fields{kfield})); end
            
            % thresholding based on single compartment R2*
            [R2s0,~]    = this.R2star_trapezoidal(abs(data),this.te);
            mask        = and(mask,R2s0>this.thres_R2star);

            % DIMWI
            if ~fitting.DIMWI.isFitFreqMW || ~fitting.DIMWI.isFitFreqIW || ~fitting.DIMWI.isFitR2sEW
                % fibre fraction
                if isfield(extraData,'ff')
                    extraData.ff                        = bsxfun(@rdivide,extraData.ff,sum(extraData.ff,4));
                    mask                                = and(mask,min(~isnan(extraData.ff),[],4));
                    extraData.ff(isnan(extraData.ff))   = 0;
                else
                    error('Fibre fraction map is required for DIMWI!');
                end
                % fibre orientation
                if ~isfield(extraData,'theta')
                    if ~isfield(extraData,'fo')
                        error('Fibre orientation map is required for DIMWI!');
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

        end

        % normalise input data based on masked signal intensity at 98%
        function [img, scaleFactor] = prepare_data(this,img, mask)

            [~,S0] = this.R2star_trapezoidal(abs(img),this.te);

            scaleFactor = prctile( S0(mask), 98);

            img = img ./ scaleFactor;

        end
        
    end

    methods(Static)

        % compute weights for optimisation
        function w = compute_optimisation_weights(data,fitting)
        % 
        % Output
        % ------
        % w         : ND signal masked wegiths that matches the arrangement in masked data later on
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
            
            % separate real/imaginary parts into 6th dim
            if fitting.isComplex
                w = repmat(w,1,1,1,1,2);
            end
        end
       
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
        function fitting2 = check_set_default(fitting,data)
            % get basic fitting setting check
            fitting2 = askadam.check_set_default_basic(fitting);

            % check weighted sum of cost function
            if ~isfield(fitting,'isWeighted');      fitting2.isWeighted     = true;         end
            if ~isfield(fitting,'weightMethod');    fitting2.weightMethod   = '1stecho';    end
            if ~isfield(fitting,'weightPower');     fitting2.weightPower    = 1;            end
            
            % check hollow cylinder fibre model parameters
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitFreqMW');  fitting2.DIMWI.isFitFreqMW  = true; end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitFreqIW');  fitting2.DIMWI.isFitFreqIW  = true; end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitR2sEW');   fitting2.DIMWI.isFitR2sEW   = true; end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitIWF');     fitting2.DIMWI.isFitIWF     = true; end

            % get customised fitting setting check
            if ~isfield(fitting,'regmap');      fitting2.regmap = 'MWF'; end
            if ~isfield(fitting,'start');       fitting2.start  = 'prior'; end

            if ~isfield(fitting,'isComplex');   fitting2.isComplex = true; end
            if isreal(data);                    fitting.isComplex = false;  end

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