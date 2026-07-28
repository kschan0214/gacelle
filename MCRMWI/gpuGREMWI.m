classdef gpuGREMWI < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 22 July 2024
% Date modified: 22 September 2024
% Date modified: 17 June 2026

    properties (GetAccess = public, SetAccess = protected)
    % ===== MODEL PARAMETER CONTRACT =====
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
    % noise     : noise
    %
    % modelParams{k} <-> ub(k) <-> lb(k) <-> startPoint(k) <-> step(k)
    % These five arrays MUST stay the same length and index-aligned.
    % Mutate only as a set, via updateProperty() - never assign into a
    % single element from outside the class, or these will desync.
    %
    % 'noise' is solver-conditional (mcmc only) and is kept LAST so that
    % updateProperty() can strip it by name without hardcoding an index,
    % and fit() (~line 521) can locate it via this.modelParams{end}. Any
    % future solver-conditional parameter should likewise go last.
        modelParams     = { 'S0';   'MWF';  'IWF';  'R2sMW';'R2sIW';'R2sEW'; 'freqMW';'freqIW';'dfreqBKG';'dpini';'noise'};
        ub              = [    2;     0.3;      1;      200;     50;     50;     0.25;    0.05;       0.4;   pi/2;    0.1];
        lb              = [ 1e-8;    1e-8;   1e-8;       50;      2;      2;    -0.05;    -0.1;      -0.4;  -pi/2;  0.001];
        startPoint      = [    1;     0.1;    0.6;      100;     15;     21;     0.04;       0;         0;      0;  0.005];
        step            = [  0.1;   0.015;   0.05;       10;    3.5;    3.5;     0.02;     0.01;     0.05;  0.075;  0.005];
    end

    properties
    % ===== USER-TUNABLE OPTIONS =====
    % Freely settable by users before fitting; no coupling between these.
        thres_similarity    = 0.1; 
        thres_impossible    = 0.1;
        thres_bkg           = 0.01;

        thres_R2star = 2;

        seed = 48463;   % for reproducible random number generation

        epsilon = utils.epsilon;

        freq_EW = 0;
    end

    properties (GetAccess = public, SetAccess = protected)
    % ===== ACQUISITION PARAMETERS =====
    % Set once in the constructor from user-provided acquisition info.
    % Read-only after construction.
        te;

        B0      = 3;            % T
        x_i     = -0.1;         % ppm
        x_a     = -0.1;         % ppm
        E       = 0.02;         % ppm
        rho_mw  = 0.36/0.86;    % ratio
        B0dir   = [0;0;1];      % unit vector [x,y,z]
    end

    properties (Constant)
            gyro = 42.57747892;
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

            % property change in related to solver
            if ~strcmpi(fitting.solver,'mcmc')
                idx = find(ismember(this.modelParams,'noise'));
                this.modelParams(idx)       = [];
                this.lb(idx)                = [];
                this.ub(idx)                = [];
                this.startPoint(idx)        = [];
                this.step(idx)              = [];
            end

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

            disp('===========');
            disp('GRE-(DI)MWI');
            disp('===========');
            
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

            % compute rotationally invariant signal if needed
            [data, mask, extraData, scaleFactor] = this.prepare_data(data,mask,extraData,fitting);

            % convert datatype to single
            data    = single(data);
            mask    = mask >0;

            %%%%%%%%%%%%%%%% Step 2: Memory management %%%%%%%%%%%%%%%%
            
            % --- [Experimental] estimate memory usage using a small batch of data size ---
            % this method tends to be more conservative than the actual memory ussage
            [seg,NSegment] = utils.find_optimal_segment_3D(this, data, mask, fitting, extraData);

            % parameter estimation
            out = [];
            for kseg = 1:NSegment
                
                if NSegment > 1
                    fprintf('Running #Segment = %d/%d \n',kseg,NSegment);
                    disp   ('------------------------')
                end
    
                % divide the data; fitRange includes halo slices (if any), ownedRange
                % is what this segment is responsible for writing back
                fitRange                        = seg(kseg).fit;
                ownedRange                      = seg(kseg).owned;
                [dataSeg, maskSeg,extraDataSeg] = this.slice_segment(data, mask, fitRange, extraData);

                % run fitting
                [outSeg] = this.fit(dataSeg,maskSeg,fitting,extraDataSeg);

                % discard halo slices from this segment's output before restoring,
                % so segment boundaries never keep voxels from a neighbour's
                % independently-converged fit (no-op when seg(kseg).fit == .owned)
                outSeg = utils.crop_segment_output(outSeg, seg(kseg));

                % restore 'out' structure from segment
                out = utils.restore_segment_structure(out,outSeg,ownedRange,kseg);

            end
            out.mask = mask;
            %%%%%%%%%%%%%%%% End Step 2 %%%%%%%%%%%%%%%%
            
            % save the estimation results if the output filename is provided
            % askadam.save_askadam_output(fitting.outputFilename,out)
            switch fitting.solver
                case 'askadam'

                    out.min.S0      = out.min.S0 * scaleFactor; % undo scaling
                    out.final.S0    = out.final.S0 * scaleFactor; % undo scaling

                    askadam.save_askadam_output(fitting.outputFilename,out)
                case 'mcmc'

                    % rescale M0
                    for k = 1:numel(fitting.metric)
                        out.(fitting.metric{k}).S0 = out.(fitting.metric{k}).S0 *scaleFactor;
                    end
                    out.posterior.S0 = out.posterior.S0 *scaleFactor;
                    mcmc.save_mcmc_output(fitting.outputFilename,out)
            end

        end

        % Data fitting function, can be 2D (voxel) or 4D (image-based)
        function [out] = fit(this,data,mask,fitting,extraData)
        %
        % Input
        % -----------
        % data      : S0 normalised 4D dwi images, [x,y,slice,diffusion], 4th dimension corresponding to [Sl0_b1,Sl0_b2,Sl2_b1,Sl2_b2, etc.]; the order of bval must match the order in the constructor gpuNEXI
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
            
            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            w = this.compute_optimisation_weights(data,fitting); % This is a customised funtion

            % split data into real and imaginary parts for complex-valued data
            if fitting.isComplex; data = cat(5,real(data),imag(data)); end

            % 2.2 display optimisation algorithm parameters
            this.display_algorithm_info(fitting)

            extraData   = utils.masking_ND2GD_preserve_struct(extraData,mask) ;

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            % 2.3 askAdam optimisation main
            switch fitting.solver
                case 'askadam'

                    out         = askadam().optimisation(data, mask, w, pars0, fitting, @this.FWD, fitting, extraData);
                    % out         = askadamObj.optimisation( dwi, mask, w, pars0, fitting, @this.FWD, fitting.lmax, fitting.solver);

                case 'mcmc'
                    fitting.xStepSize = this.step;

                    % 3.1. initial global optimisation
                    out         = mcmc().optimisation(data, mask, w, pars0, fitting, @this.FWD, fitting, extraData);
                    
            end

            disp('The process is completed.')
            
            % clear GPU
            reset(gpuDevice)
            
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

            dims = size(data,1:3);

            % initiate starting point of all parameters
            pars0 = utils.initialise_x0(dims,this.modelParams,this.startPoint);
            
            disp('Estimate starting points based on hybrid fixed points/prior information ...')

            % update S0
            [R2s,S0]  = this.R2star_trapezoidal(abs(data),this.te);
            S0(isnan(S0)) = 0; S0(isinf(S0)) = 0; S0(S0<0) = 0;
            pars0.S0 = single(S0);

            % update R2*IW
            idx = find(ismember(this.modelParams,'R2sIW'));
            R2sIW = R2s - 3;
            R2sIW(isnan(R2sIW)) = single(this.startPoint(idx)); R2sIW(isinf(R2sIW)) = single(this.startPoint(idx)); 
            R2sIW = single(max(min(R2sIW,this.ub(idx)-2),this.lb(idx)+2));  % avoid boundaries
            pars0.R2sIW = single(R2sIW);

            % update R2*EW
            idx = find(ismember(this.modelParams,'R2sEW'));
            if ~isempty(idx)
                % if R2*EW is a free parameter then set it
                R2sEW = R2s + 3;
                R2sEW(isnan(R2sEW)) = single(this.startPoint(idx)); R2sEW(isinf(R2sEW)) = single(this.startPoint(idx)); 
                R2sEW = single(max(min(R2sEW,this.ub(idx)-2),this.lb(idx)+2));  % avoid boundaries
                pars0.R2sEW = single(R2sEW);
            end

        end

        %% Signal related functions

        % Forward model to generate GRE-MWI signal
        function [s] = FWD(this, pars, fitting, extraData)


            TE =  permute(this.te, [2 3 4 1] );              % TE always on 4th dim

            S0   = pars.S0;
            mwf  = pars.MWF;
            if fitting.DIMWI.isFitIWF; iwf  = pars.IWF; else; iwf = extraData.IWF; end
            r2sMW   = pars.R2sMW;
            r2sIW   = pars.R2sIW;

            % external effects
            if ~fitting.isComplex % magnitude fitting
                freqBKG = 0;                          
                pini    = 0;
            else    % other fittings
                freqBKG = pars.dfreqBKG + extraData.freqBKG; 
                pini    = pars.dpini + extraData.pini;
            end
        
            %%%%%%%%%%%%%%%%%%%% Compartmental Signals %%%%%%%%%%%%%%%%%%%%
            if isfield(fitting,'solver') && strcmpi(fitting.solver, 'mcmc')
                [S0MW,S0IW,S0EW] = arrayfun(@gremwi_S0_compartments,S0,mwf,iwf);
            else
                [S0MW,S0IW,S0EW] = gremwi_S0_compartments(S0,mwf,iwf);
            end

            %%%%%%%%%%%%%%%%%%%% DIMWI related operations %%%%%%%%%%%%%%%%%%%%
            
            % if use HCFM to derive either freqMW|freqIW|R2*EW, then computee g-ratio
            if ~fitting.DIMWI.isFitFreqMW || ~fitting.DIMWI.isFitFreqIW || ~fitting.DIMWI.isFitR2sEW
                hcfm_obj = HCFM(this.te,this.B0);

                % g-ratio 
                if isfield(fitting,'solver') && strcmpi(fitting.solver, 'mcmc')
                    g = arrayfun(@hcfm_gratio,abs(S0IW),abs(S0MW)/this.rho_mw);
                else
                    g = hcfm_gratio(abs(S0IW),abs(S0MW)/this.rho_mw);
                end
                g = max(g, this.epsilon);
            end
            
            % extra decay on extracellular water estimated by HCFM 
            if ~fitting.DIMWI.isFitR2sEW
                
                % assume extracellular water has the same T2* as intra-axonal water
                r2sEW   = r2sIW;
                % fibre volume fraction
                if isfield(fitting,'solver') && strcmpi(fitting.solver, 'mcmc')
                    fvf = arrayfun(@hcfm_fibre_volume_fraction,abs(S0IW),abs(S0EW),abs(S0MW)/this.rho_mw);
                else
                    fvf = hcfm_fibre_volume_fraction(abs(S0IW),abs(S0EW),abs(S0MW)/this.rho_mw);
                end

                % signal dephase in extracellular water due to myelin sheath, Eq.[A7]
                decayEW = hcfm_obj.DephasingExtraaxonal(fvf,g,this.x_i,this.x_a,extraData.theta);
                decayEW = permute(decayEW,[4 2 3 1 5]); % fibre in 5th dimension

            else
                decayEW = 0;
                r2sEW   = pars.R2sEW;
            end

            % compute frequency shifts given theta
            if ~fitting.DIMWI.isFitFreqMW 

                % in ppm
                freqMW = hcfm_obj.FrequencyMyelin(this.x_i,this.x_a,g,extraData.theta,this.E) / (this.B0*this.gyro);
            else

                freqMW  = pars.freqMW;
            end

            if ~fitting.DIMWI.isFitFreqIW 

                % in ppm
                freqIW = hcfm_obj.FrequencyAxon(this.x_a,g,extraData.theta) / (this.B0*this.gyro);

            else

                freqIW  = pars.freqIW;

            end

            freqEW      = this.freq_EW;
            S0IEW_phase = 0;

            if isfield(fitting,'solver') && strcmpi(fitting.solver, 'mcmc')
                [Sreal,Simag] = arrayfun(@compute_gremwi_signal,S0MW,S0IW,S0EW,r2sMW,r2sIW,r2sEW,freqMW,freqIW,freqEW,freqBKG,pini,decayEW,extraData.ff,TE,this.B0,this.gyro,S0IEW_phase);
            else
                [Sreal,Simag] = compute_gremwi_signal(S0MW,S0IW,S0EW,r2sMW,r2sIW,r2sEW,freqMW,freqIW,freqEW,freqBKG,pini,decayEW,extraData.ff,TE,this.B0,this.gyro,S0IEW_phase);
            end
            % weighted sum all fibre for DIMWI
            Sreal = sum(Sreal,5);
            Simag = sum(Simag,5);

            if ~fitting.isComplex
                s = sqrt(Sreal.^2 + Simag.^2);
            else
                s = cat(5,Sreal,Simag);
            end

            % vectorise to match maksed measurement data
            s = utils.reshape_ND2GD(s,[]);

            if isfield(fitting,'solver') && strcmpi(fitting.solver, 'mcmc')
                % reshape s for GW
                if ~isempty(fitting)
                    if strcmpi(fitting.algorithm,'ensemble')
                        s = reshape(s, [size(s,1) size(s,2)/fitting.Nwalker fitting.Nwalker]);
                    end
                end
            end


        end
        
        %% Utilities

        % validate extra data
        function [extraData,mask] = validate_data(this,data,extraData,mask,fitting)

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
                    extraData.ff                        = permute(extraData.ff,[1 2 3 5 4]);
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
                            theta(:,:,:,:,kfo) = this.AngleBetweenV1MapAndB0(fo(:,:,:,:,kfo),this.B0dir);
                        end
                        extraData.theta = single(theta);
                        extraData = rmfield(extraData,"fo");
                    end
                else
                    extraData.theta = permute(extraData.theta,[1 2 3 5 4]);
                end
            else
                extraData.theta = zeros(dims,'single');
                extraData.ff    = ones(dims,'single');
            end

        end

        % normalise input data based on masked signal intensity at 98%
        function [data, mask, extraData, scaleFactor] = prepare_data(this,data,mask,extraData,fitting)

            % make sure input data are valid
            [extraData,mask] = this.validate_data(data,extraData,mask,fitting);

            [~,S0] = this.R2star_trapezoidal(abs(data),this.te);

            scaleFactor = prctile( S0(mask), 98);

            data = data ./ scaleFactor;

            % mask sure no nan or inf
            [data,mask] = utils.remove_img_naninf(data,mask);

        end
        
        % segment data based on slice
        function [dataSeg, maskSeg, extraDataSeg] = slice_segment(this, data, mask, slice, extraData)

            dataSeg     = data(:,:,slice,:,:,:,:,:,:);
            maskSeg     = mask(:,:,slice);
            if ~isempty(extraData)
                fields      = fieldnames(extraData); 
                for kfield = 1:numel(fields)
                    extraDataSeg.(fields{kfield}) = extraData.(fields{kfield})(:,:,slice,:,:,:,:,:,:,:,:); 
                end
            else                                                    
                extraDataSeg = [];                 
            end

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
            
            if ~isfield(fitting,'solver');      fitting.solver = 'askadam';        end

            % get basic fitting setting check
            if strcmpi(fitting.solver,'mcmc')

                % mcmc
                fitting2 = mcmc.check_set_default_basic(fitting);

            else

                % askadam
                fitting2 = askadam.check_set_default_basic(fitting);
                % get customised fitting setting check
                if ~isfield(fitting,'regmap');      fitting2.regmap = 'MWF'; end
            end

            % get customised fitting setting check
            if ~isfield(fitting,'weightMethod');        fitting2.weightMethod   = '1stecho';        end
            if ~isfield(fitting,'isWeighted');          fitting2.isWeighted     = true;            end
            if ~isfield(fitting,'weightPower');         fitting2.weightPower    = 1;                end
            if ~isfield(fitting,'start');               fitting2.start          = 'prior';          end
            
            % check hollow cylinder fibre model parameters
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitFreqMW');  fitting2.DIMWI.isFitFreqMW  = true; end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitFreqIW');  fitting2.DIMWI.isFitFreqIW  = true; end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitR2sEW');   fitting2.DIMWI.isFitR2sEW   = true; end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitIWF');     fitting2.DIMWI.isFitIWF     = true; end

            
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