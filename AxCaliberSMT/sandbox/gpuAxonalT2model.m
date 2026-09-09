classdef gpuAxonalT2model < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 29 September 2025
% Date modified: 

% TODO: sort b and te and input full DWI

    properties
        % default model parameters and estimation boundary
        % r     : Axon radius [um]
        % s0    : DW signal [au]    
        % k2a   : rate of axon radius induced T2 [um/s]
        % R2a   : Intrinsic neurite R2 [1/s]
        % 
        modelParams     = {  'r'; 'S0';'k2a';'R2a'};
        ub              = [    5;    5;    4; 20];
        lb              = [1e-10;    0;    0;  5];
        startPoint      = [    1;  0.1;  2.4;  8];

    end

    properties (Constant)

        epsilon = 1e-10;

    end

    properties (GetAccess = public, SetAccess = protected)
        te;

        % tissue properties
        R2c     = 1/126.97e-3;  % axonal T2 constant term [1/s] 
        rho2    = 1.16*2;       % axonal surface to volume ratio [um/s]
    end
    
    properties (GetAccess = private, SetAccess = private)
       
    end
    
    methods

        % constructuor
        function this = gpuAxonalT2model(te, tissueProperties)
        % Estimation of neurite fraction and intrinsic diffusivity using SMT with th eoption of performing compartmental T2 mapping
        % smt = gpuSMT(b, te)
        %       output:
        %           - smt: object of a fitting class
        %
        %       input:
        %           - te:       echo time               [s] (if Nt>1)
        %           - tissueProperties
        %               .R2c:   Intrinsic axonal R2 [1/s] (default:1/126.97e-3)
        %               .rho2:  coeffient of 1/r dependence [um/s] (default:1.16*2)
        %           -model: 'narrow' or 'wide' pulse
        %
        %  Authors: 
        %  Kwok-Shing Chan (kchan2@mgh.harvard.edu)
        %
            
            % sequence parameters
            % relaxation
            this.te = single(te(:));    % same length as b or scalar (if Nt==1)
            
            % user defined
            % relaxation
            if isfield(tissueProperties,'R2c');     this.R2c    = (single( R2c ));      end
            if isfield(tissueProperties,'rho2');    this.rho2   = (single( rho2 ));     end


        end

        % update properties according to lmax
        function this = updateProperty(this, fitting)

            % whether fitting CSF compartment or not
            if ~fitting.isFitR2a
                idx = find(ismember(this.modelParams,'R2a'));
                this.modelParams(idx)       = [];
                this.lb(idx)                = [];
                this.ub(idx)                = [];
                this.startPoint(idx)        = [];
            end
            % whether fitting CSF compartment or not
            if ~fitting.isFitk2a
                idx = find(ismember(this.modelParams,'k2a'));
                this.modelParams(idx)       = [];
                this.lb(idx)                = [];
                this.ub(idx)                = [];
                this.startPoint(idx)        = [];
            end

        end
    
        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('===========================================================');
            disp('Axon radius mapping based on T2 model with askadam.m solver');
            disp('===========================================================');

            disp('----------------')
            disp('Data Information');
            disp('----------------')
            fprintf('TE (ms)                                    : [%s] \n',num2str(this.te.'*1e3,' %.2f'));
            disp('----------------')

            disp('----------------')
            disp('Fixed parameters');
            disp('----------------')
            disp(['Intrinsic intra-axonal T2 (ms)               : ' num2str(1/this.R2c*1e3,'%.2f')]);
            disp(['Surface-to-volume ratio constant (ms)        : ' num2str(this.rho2,'%.2f')]);
            disp('----------------')

        end

        %% higher-level data fitting functions

        % This is a wrapper of the 'fit' function.
        % The main purpose of this function is to handle memory issue and ensure the input data is correct for 'fit'
        function  [out] = estimate(this, dwi, mask, fitting, extraData, pars0)
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
        % fitting   : fitting algorithm parameters (see fit function)
        % 
        % Output
        % -----------
        % out       : output structure contains all parameter estimation results
        % 
            % if no pars input at all (not even empty) then use prior
            if nargin < 6; pars0        = []; end
            if nargin < 5; extraData    = []; end

            % display basic info
            this.display_data_model_info;

            % get all fitting algorithm parameters 
            fitting = this.check_set_default(fitting);

            % get matrix size
            dims = size(dwi,1:3);

            %%%%%%%%%%%%%%%% Step 1: Validate all input data %%%%%%%%%%%%%%%%
            % compute rotationally invariant signal if needed
            [dwi, scaleFactor] = this.prepare_dwi_data(dwi,mask);

            % mask sure no nan or inf in data
            [dwi,mask] = utils.remove_img_naninf(dwi,mask);

            % convert datatype to single or logical
            dwi     = single(dwi);
            mask    = mask >0;
            if ~isempty(pars0); for km = 1:numel(this.modelParams); pars0.(this.modelParams{km}) = single(pars0.(this.modelParams{km})); end; end

            %%%%%%%%%%%%%%%% End Step 1 %%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%% Step 2: fit on whole data  %%%%%%%%%%%%%%%%
            % parameter estimation
            out  = this.fit(dwi,mask,fitting,pars0);

            out.mask = mask;
            % rescale S0
            out.final.S0    = out.final.S0 *scaleFactor;
            out.min.S0      = out.min.S0 *scaleFactor;
            %%%%%%%%%%%%%%%% End Step 2 %%%%%%%%%%%%%%%%

            % save the estimation results if the output filename is provided
            askadam.save_askadam_output(fitting.outputFilename,out)

        end

        % Data fitting function
        % This is a wapper of the askadam class 'fit' function
        function [out] = fit(this,dwi,mask,fitting, pars0)
        %
        % Input
        % -----------
        % dwi       : S0 normalised 4D dwi images, [x,y,slice,diffusion], 4th dimension corresponding to [Sl0_b1,Sl0_b2 etc.]; the order of bval must match the order in the constructor 
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
        % pars0     : structure variable of starting points of fitting (optional)
        % 
        % Output
        % -----------
        % out       : output structure
        %   .final      : final results
        %   .min        : results with the minimum loss metric across all iterations
        %       .loss       : loss metric      
        %
        % Kwok-Shing Chan @ MGH
        % kchan2@mgh.harvard.edu
        % Date created: 1 Oct 2025
        % Date modified:
        %
            
            % check GPU
            gpool = gpuDevice;
            
            % check image size
            dims = size(dwi,1:3);

            %%%%%%%%%%%%%%%%%%%% Step 1. Validate and parse input %%%%%%%%%%%%%%%%%%%%
            if nargin < 3 || isempty(mask); mask = ones(dims,'logical'); end % if no mask input then fit everthing
            if nargin < 4; fitting = struct(); end
            % set initial tarting points
            if nargin < 5; pars0 = []; % no initial starting points
            else
                if ~isempty(pars0); for km = 1:numel(this.modelParams); pars0.(this.modelParams{km}) = single(pars0.(this.modelParams{km})); end; end
            end

            % get all fitting algorithm parameters 
            fitting                 = this.check_set_default(fitting);
            % determine fitting parameters
            this                    = this.updateProperty(fitting);
            fitting.modelParams     = this.modelParams;
            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(fitting.modelParams)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(fitting.modelParams)); end

            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights
            % lmax    = 0;
            % w       = this.compute_optimisation_weights(mask,fitting.lossFunction,lmax); % This is a customised funtion
            w       = this.compute_optimisation_weights(mask); % This is a customised funtion

            % 2.2 estimate prior if needed
            if isempty(pars0);  pars0 = this.determine_x0(dwi,mask,fitting); end

            % 2.3 askAdam optimisation main
            askadamObj  = askadam(); 
            out         = askadamObj.optimisation(dwi, mask, w, pars0, fitting, @this.FWD);

            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            disp('The estimation is completed.');
            
            % clear GPU
            reset(gpool)

        end

        %% Data preparation

        % compute rotationally invariant DWI signal if necessary
        % TODO
        function [dwi, scaleFactor] = prepare_dwi_data(this,dwi,mask)
            
            [~,m0] = this.R2_lsq(dwi,this.te,mask);

            scaleFactor = prctile(m0(mask>0),95);

            dwi = dwi./ scaleFactor;
            
        end

        % compute weights for optimisation
        % function w = compute_optimisation_weights(this,mask,lossFunction,lmax)
        function w = compute_optimisation_weights(this,mask)
        % 
        % Output
        % ------
        % w         : N-D signal masked wegiths
        %
            dims = size(mask,1:3);

            % w = ones([dims,numel(this.te)]);\
            w = single(repmat(mask,1,1,1,numel(this.te)));

            % % lmax dependent weights
            % l = 0:2:lmax;
            % w = zeros([dims numel(this.b)*numel(l)],'single');
            % for kl = 1:(lmax/2+1)
            %     for kb = 1:numel(this.b)
            %         w(:,:,:,(kl-1)*numel(this.b)+kb) = 1/ (2*l(kl)+1);
            %     end
            % end
            % % if L1 then take square root
            % if strcmpi(lossFunction,'l1')
            %     w = sqrt(w);
            % end
            % w = w ./ max(w(:));
            
        end

        %%%%% Prior estimation related functions %%%%%

        % determine how the starting points will be set up
        function x0 = determine_x0(this,y,mask,fitting) 

            disp('---------------');
            disp('Starting points');
            disp('---------------');

            dims = size(mask,1:3);

            if ischar(fitting.start)
                switch lower(fitting.start)
                    case 'likelihood'
                        % using maximum likelihood method to estimate starting points
                        x0 = this.estimate_prior(y,mask);
                        % R2a and k2a are global constants
                        % if any(ismember(this.modelParams,'R2a')); x0.R2a = this.R2c;    end
                        % if any(ismember(this.modelParams,'k2a')); x0.k2a = this.rho2;   end
    
                    case 'default'
                        % use fixed points
                        fprintf('Using default starting points for all voxels at [%s]: [%s]\n', cell2str(this.modelParams),replace(num2str(this.startPoint(:).',' %.2f'),' ',','));
                        x0 = utils.initialise_x0(dims,this.modelParams,this.startPoint);
                        % R2a and k2a are global constants
                        if any(ismember(this.modelParams,'R2a')); x0.R2a = this.R2c;    end
                        if any(ismember(this.modelParams,'k2a')); x0.k2a = this.rho2;   end

                end
            else
                % user defined starting point
                x0 = fitting.start(:);
                fprintf('Using user-defined starting points for all voxels at [%s]: [%s]\n',cell2str(this.modelParams),replace(num2str(x0(:).',' %.2f'),' ',','));
                x0 = utils.initialise_x0(dims,this.modelParams,x0);
                if any(ismember(this.modelParams,'R2a')); x0.R2a = this.R2c;    end
                if any(ismember(this.modelParams,'k2a')); x0.k2a = this.rho2;   end

            end

            % make sure the input is bounded
            x0 = askadam.set_boundary(x0,fitting.ub,fitting.lb);
            
            fprintf('Estimation lower bound [%s]: [%s]\n',      cell2str(this.modelParams),replace(num2str(fitting.lb(:).',' %.2f'),' ',','));
            fprintf('Estimation upper bound [%s]: [%s]\n',      cell2str(this.modelParams),replace(num2str(fitting.ub(:).',' %.2f'),'  ',','));
            ('---------------');
        end

        % using maximum likelihood method to estimate starting points
        function pars0 = estimate_prior(this,dwi,mask)
        % Estimation starting points for NEXI using likehood method

            disp('Estimate starting points based on likelihood ...')

            [r2,s0]  = this.R2_lsq(dwi,this.te,double(mask));

            % convert to radius
            r_max       = 20; % um, upper bound
            r           = this.rho2./(r2 - this.R2c); % inverse Eq.[1]
            r(r<0)      = 0;
            r(r>r_max)  = r_max;
            r           = r .* mask;

            s0(isnan(s0))       = this.lb(2);
            s0(isinf(s0))       = this.lb(2);
            s0(s0<0)            = 0;
            s0(s0>this.ub(2))   = this.ub(2);

            %  initiate pars0 with the same order as modelParams
            for k = 1:numel(this.modelParams)
                pars0.(this.modelParams{k}) = [];
            end

            pars0.r     = r;
            pars0.S0    = s0;
            % global constant for the data
            if any(ismember(this.modelParams,'k2a'))
                pars0.k2a  = this.rho2;
            end
            if any(ismember(this.modelParams,'R2a'))
                pars0.R2a  = this.R2c;
            end

        end

        %%  Signal related functions
        
        % Forward model
        function s = FWD(this, pars)

            % minimal fitting parameters
            r   = pars.r;
            S0  = pars.S0;
            if isfield(pars,'R2a')
                R2a = pars.R2a;
            else
                R2a = this.R2c;
            end
            if isfield(pars,'k2a')
                k2a = pars.k2a;
            else
                k2a = this.rho2;
            end

            % intra-axonal signal
            s = this.signal_axonal_T2(this.te,S0,r, R2a, k2a) ;
                
        end

    end

    methods(Static)

        %% Utility
        %%%%%%%%%% Compartmental signal
        % restricted stick
        function S = signal_axonal_T2(te,s0,r, R2a, k2a)

            S = s0.* exp(-te.*( R2a + k2a./r));

        end

        function [r2,m0] = R2_lsq(img,te,mask)

            img = double(img);
            te  = double(te);

            % set range of R2 and T2
            minT2s      = min(te)/20;
            maxT2s      = max(te)*20;
            ranger2     = [1/maxT2s, 1/minT2s];
            
            [nx,ny,nz,nt] = size(img);
            
            x       = ones(nt,2);
            x(:,2)  = -te(:);
            y       = permute(log(abs(img)),[4 1 2 3]);
            
            y       = reshape(y,[size(y,1) numel(y)/size(y,1)]);
            b       = x\y;
            r2      = reshape( b(2,:),nx,ny,nz) .* mask;
            m0      = exp(reshape( b(1,:),nx,ny,nz)) .* mask;
            
            r2(r2>max(ranger2)) = max(ranger2);
            r2(r2<min(ranger2)) = min(ranger2);

            m0(m0<0) = 0;

        end

        %%%%%%%%%%%%%%
        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default(fitting)
            % get basic fitting setting check
            fitting2 = askadam.check_set_default_basic(fitting);

            % get customised fitting setting check
            if ~isfield(fitting,'regmap');      fitting2.regmap     = {'r'};            end
            if ~isfield(fitting,'start');       fitting2.start      = 'likelihood';     end
            if ~isfield(fitting,'isFitR2a');    fitting2.isFitR2a   = false;            end
            if ~isfield(fitting,'isFitk2a');    fitting2.isFitk2a   = false;            end

            if ~iscell(fitting2.regmap)
                fitting2.regmap = cellstr(fitting2.regmap);
            end

        end
    
    end

end