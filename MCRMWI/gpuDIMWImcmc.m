classdef gpuDIMWImcmc < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% AxCaliberSMT model parameter estimation based on MCMC
% Date created: 22 March 2024 
% Date modified: 14 June 2024

    properties (Constant)
            gyro = 42.57747892;
    end

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

        model_params    = { 'S0';   'MWF';  'IWF';  'R2sMW';'R2sIW';'R2sEW'; 'freqMW';'freqIW';'dfreqBKG';'dpini'; 'noise'};
        ub              = [    2;     0.3;      1;      300;     40;     40;      0.3;     0.1;       0.4;   pi/2;     0.1];
        lb              = [    0;       0;      0;       40;      1;      1;        0;    -0.1;      -0.4;  -pi/2;    0.01];
        step            = [  0.1;   0.015;   0.05;        5;      1;      1;     0.01;    0.01;      0.01;   0.05;   0.005];
        startpoint      = [    1;     0.1;    0.8;      100;     15;     21;     0.04;       0;         0;      0;    0.05];
        % model_params    = { 'S0';   'MWF';  'IWF';  'T2sMW';'T2sIW';'T2sEW'; 'freqMW';'freqIW';'dfreqBKG';'dpini'; 'noise'};
        % ub              = [    2;     0.3;      1;       25;    200;    200;      0.3;     0.1;       0.4;   pi/2;     0.1];
        % lb              = [    0;       0;      0;        3;     25;     25;        0;    -0.1;      -0.4;  -pi/2;    0.01];
        % step            = [ 0.05;    0.01;   0.01;      0.5;      1;      1;     0.01;    0.01;     0.005;   0.01;   0.005];
        % startpoint      = [    1;     0.1;    0.8;       10;     64;     48;     0.04;       0;         0;      0;    0.05];

    end

    properties (Constant = true, Access = protected)
    end
    
    properties (GetAccess = public, SetAccess = protected)

        B0      = 3;            % T
        x_i     = -0.1;         % ppm
        x_a     = -0.1;         % ppm
        E       = 0.02;         % ppm
        rho_mw  = 0.36/0.86;    % ratio
        B0dir   = [0;0;1];      % unit vector [x,y,z]

        te
        
    end
    
    properties (GetAccess = private, SetAccess = private)
       
    end
    
    methods (Access = public)

        function this = gpuDIMWImcmc(te,fixed_params)
        % gpuDIMWImcmc GRE-MWI
        % smt = gpuDIMWImcmc(te,fixed_params)
        %       output:
        %           - smt: object of a fitting class
        %
        %       input:
        %           - te: Echo time [s]
        %           - fixed_params: parameter to be fixed
        %               - x_i   : isotropic susceptibility of myelin [ppm]
        %               - x_a   : anisotropic susceptibility of myelin [ppm]
        %               - E     : exchange induced frequency shift [ppm]
        %               - rho_mw: myelin water proton ratio
        %               - B0    : main magnetic field strength [T]
        %               - B0dir : main magnetic field direction, [x,y,z]
        % 
        %
        %       usage:
        %           this = gpuDIMWImcmc(te,fixed_params)
        %
        %  Authors: 
        %  Kwok-Shing Chan (kchan2@mgh.harvard.edu)
        %  Copyright (c) 2022 Massachusetts General Hospital
            
            this.te     = ( single(te(:)) );
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
            end
        end

        % determining the fitting parameters
        function this = updateProperty(this, fitting)

            if fitting.isComplex == 0
                for kpar = {'dfreqBKG','dpini'}
                    idx = find(ismember(this.model_params,kpar));
                    this.model_params(idx)    = [];
                    this.lb(idx)              = [];
                    this.ub(idx)              = [];
                    this.startpoint(idx)      = [];
                    this.step(idx)            = [];
                end
            end

            % DIMWI
            if fitting.DIMWI.isFitFreqIW == 0
                idx = find(ismember(this.model_params,'freqIW'));
                this.model_params(idx)    = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startpoint(idx)      = [];
                this.step(idx)            = [];
            end

            if fitting.DIMWI.isFitFreqMW == 0
                idx = find(ismember(this.model_params,'freqMW'));
                this.model_params(idx)    = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startpoint(idx)      = [];
                this.step(idx)            = [];
            end

            if fitting.DIMWI.isFitVic == 0
                idx = find(ismember(this.model_params,'IWF'));
                this.model_params(idx)    = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startpoint(idx)      = [];
                this.step(idx)            = [];
            end

            if fitting.DIMWI.isFitR2sEW == 0
                idx = find(ismember(this.model_params,'T2sEW'));
                this.model_params(idx)    = [];
                this.lb(idx)              = [];
                this.ub(idx)              = [];
                this.startpoint(idx)      = [];
                this.step(idx)            = [];
            end

        end

        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('=======================================================');
            disp('GRE-(DI)MWI with Markov Chain Monte Carlo (MCMC) solver');
            disp('=======================================================');
            
            disp('----------------')
            disp('Data Information');
            disp('----------------')
            fprintf('Echo time, TE (ms)                     : [%s] \n',num2str((this.te*1e3).',' %.2f'));
            
            disp('---------------------')
            disp('Parameter to be fixed')
            disp('---------------------')
            disp(['Field strength (T)                       : ' num2str(this.B0)]);
            disp(['Relative myelin water density            : ' num2str(this.rho_mw)]);
            disp(['Myelin isotropic susceptibility (ppm)    : ' num2str(this.x_i)]);
            disp(['Myelin anisotropic susceptibility (ppm)  : ' num2str(this.x_a)]);
            disp(['Exchange term (ppm)                      : ' num2str(this.E)]);

            fprintf('\n')

        end

        % Perform AxCaliber model parameter estimation based on MCMC across the whole dataset   
        function [out] = estimate(this, data, mask, extraData, fitting)
        % Input data are expected in multi-dimensional image
        % 
        % Input
        % -----------
        % data      : 4D MEGRE, [x,y,z,t]
        % mask      : 3D signal mask, [x,y,z]
        % extradata : extra DWI protocol parameters
        %   .freqBKG: 3D initial estimation of total field [Hz] (highly recommended)
        %   .pini   : 3D initial estimation of B1 offset [rad]  (highly recommended)
        %   .ff     : 3D/4D fibre fraction map, [x,y,z,nF] (for GRE-DIMWI only)
        %   .theta  : 3D/4D angle between B0 and fibre orientation, [x,y,z, nF] (for GRE-DIMWI only)
        % fitting   : fitting algorithm parameters
        %   .iteration  : number of MCMC iterations
        %   .interval   : interval of MCMC sampling
        %   .method     : method to compute the parameters, 'mean' (default)|'median'
        %   .DIMWI
        %       .isFitFreqMW    : fit myelin water frequency or not, 'true' (default)|'false'
        %       .isFitFreqIW    : fit intracellular water frequency or not, 'true' (default)|'false'
        %       .isFitR2sEW     : fit extracellular water R2* or not, 'true' (default)|'false'
        %       .isFitVic       : fit intra-/extra cellular volume ratio or not, 'true' (default)|'false'
        % 
        % Output
        % -----------
        % out       : output structure contains all MCMC results
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

            this = this.updateProperty(fitting);

            [extraData] = this.validate_data(data,extraData);

            % compute rotationally invariant signal if needed
            [data, scaleFactor] = this.prepare_data(data,mask);

            % vectorise data, 1st dim: b-value*lmax; 2nd dim: voxels
            data = this.vectorise_4Dto2D(data,mask).';
            fields = fieldnames(extraData);
            for kfield = 1:numel(fields); extraData.(fields{kfield}) = gpuArray(single( this.vectorise_4Dto2D(extraData.(fields{kfield}),mask).' )); end
            
            % MCMC main
            [x_m, x_dist] = this.fit(data, fitting, extraData);

            % export results to organise output structure
            out = mcmc.res2out(x_m,x_dist,this.model_params,mask);
            % rescaling
            out.expected.S0     = out.expected.S0 * scaleFactor;
            out.posterior.S0    = out.posterior.S0 * scaleFactor;

            out.expected.dfreqBKG   = out.expected.dfreqBKG * this.B0*this.gyro;
            out.expected.freqIW     = out.expected.freqIW * this.B0*this.gyro;
            out.expected.freqMW     = out.expected.freqMW * this.B0*this.gyro;
    
            % for k = 1:length(this.model_params); eval([this.model_params{k} ' = out.expected.' this.model_params{k} ';']); end
            
            % save the estimation results if the output filename is provided
            mcmc.save_mcmc_output(fitting.output_filename,out)

        end
        
        % Perform parameter estimation using MCMC solver
        function [xExpected,xPosterior] = fit(this, y, fitting, extraData)
        % Input
        % -----
        % y         : measurements, 1st dim: b-value; 2nd dim: voxels
        % fitting   : fitting algorithm settings, see above
        %
        % Output
        % ------
        % xExpected : expected values of the model parameters
        % xPosterior: posterior distribution of MCMC
        %
   
            % Step 0: display basic messages
            mcmc.display_basic_algorithm_parameters(fitting);
            % additional message(s)
            if ischar(fitting.start); disp(['Starting points   : ', fitting.start ]); end; fprintf('\n');
            
            % Step 1: prepare input data
            % set starting points
            x0 = gpuArray(single(this.determine_x0(y,fitting)));

            % Step size on each iteration
            xStepsize =  gpuArray(single(this.step));

            % Step 2: parameter estimation
            [xExpected,xPosterior] = mcmc.metropilis_hastings(y,x0,xStepsize,fitting,@this.FWD,fitting, extraData);
            
        end

%% Signal generation
        
        % FWD signal model
        function S = FWD(this, pars, fitting, extraData)

            %%%%%%%%%%%%%%%%%%%% get estimation parameters %%%%%%%%%%%%%%%%%%%%
            S0      = pars(1,:);
            mwf     = pars(2,:); n = 3;

            if fitting.DIMWI.isFitVic
                iwf  = pars(n,:); n = n+1;
            else
                iwf = extraData.iwf;
            end
            r2sMW   = pars(n,:); n = n+1;
            r2sIW   = pars(n,:); n = n+1;

            if fitting.DIMWI.isFitR2sEW
                r2sEW  = pars(n,:); n = n+1;
            end
            if fitting.DIMWI.isFitFreqMW
                freqMW  = pars(n,:); n = n+1;
            end
            if fitting.DIMWI.isFitFreqIW
                freqIW  = pars(n,:); n = n+1;
            end
            % external effects
            if ~fitting.isComplex % magnitude fitting
                freqBKG = 0;                          
                pini    = 0;
            else    % other fittings
                freqBKG = pars(n,:) + extraData.freqBKG;  n = n + 1;
                pini    = pars(n,:) + extraData.pini;
            end

            %%%%%%%%%%%%%%%%%%%% DIMWI related operations %%%%%%%%%%%%%%%%%%%%
            % Compartmental Signals
            S0MW = S0 .* mwf;
            S0IW = S0 .* (1-mwf) .* iwf;
            S0EW = S0 .* (1-mwf) .* (1-iwf);

            if ~fitting.DIMWI.isFitFreqMW || ~fitting.DIMWI.isFitFreqIW || ~fitting.DIMWI.isFitR2sEW
                hcfm_obj = HCFM(this.te,this.B0);

                % derive g-ratio 
                g = hcfm_obj.gratio(abs(S0IW),abs(S0MW)/this.rho_mw);

            end
            
           % extra decay on extracellular water estimated by HCFM 
           if ~fitting.DIMWI.isFitR2sEW
                
                % assume extracellular water has the same T2* as intra-axonal water
                r2sEW   = r2sIW;

                fvf     = hcfm_obj.FibreVolumeFraction(abs(S0IW),abs(S0EW),abs(S0MW)/this.rho_mw);
                % signal dephase in extracellular water due to myelin sheath, Eq.[A7]
                decayEW = hcfm_obj.DephasingExtraaxonal(fvf,g,this.x_i,this.x_a,extraData.theta);

                if ismatrix(extraData.theta)
                    decayEW = permute(decayEW,[1,2,4,5,3]);
                else
                    decayEW = permute(decayEW,[1,2,3,5,4]);
                end
           else
                % decayEW    = zeros([length(this.te),size(S0,2)]);
                decayEW = 0;
           end

           % determine the source of compartmental frequency shifts
            if ~fitting.DIMWI.isFitFreqMW || ~fitting.DIMWI.isFitFreqIW
               
                % compute frequency shifts given theta
                if ~fitting.DIMWI.isFitFreqMW 
                    freqMW = hcfm_obj.FrequencyMyelin(this.x_i,this.x_a,g,extraData.theta,this.E);
                    freqMW = permute(freqMW,[1 2 3 5 4]);
                end
                if ~fitting.DIMWI.isFitFreqIW 
                    freqIW = hcfm_obj.FrequencyAxon(this.x_a,g,extraData.theta);
                    freqIW = permute(freqIW,[1 2 3 5 4]);
                end
            end

            extraData.ff = permute(extraData.ff,[3 2 1]);
           
            freqEW = 0;

            %%%%%%%%%%%%%%%%%%%% Forward model %%%%%%%%%%%%%%%%%%%%
            [Sreal,Simag] = arrayfun(@model_DIMWI,this.te, this.B0, this.gyro, S0MW, S0IW, S0EW, r2sMW, r2sIW, r2sEW, decayEW, freqMW+freqBKG, freqIW+freqBKG, freqEW+freqBKG, pini);
            Sreal = sum(Sreal.*extraData.ff,3);
            Simag = sum(Simag.*extraData.ff,3);

            % Sreal = sum((   S0MW .* exp(-this.te ./ t2sMW) .* cos(this.te .* 2*pi*(freqMW+freqBKG)*this.B0*this.gyro + pini) + ...
            %                 S0IW .* exp(-this.te ./ t2sIW) .* cos(this.te .* 2*pi*(freqIW+freqBKG)*this.B0*this.gyro + pini) + ...
            %                 S0EW .* exp(-this.te ./ t2sEW) .* cos(this.te .* 2*pi*(freqEW+freqBKG)*this.B0*this.gyro + pini) .* exp(decayEW) ).*extraData.ff,3);
            % 
            % Simag = sum((   S0MW .* exp(-this.te ./ t2sMW) .* sin(this.te .* 2*pi*(freqMW+freqBKG)*this.B0*this.gyro + pini) + ...
            %                 S0IW .* exp(-this.te ./ t2sIW) .* sin(this.te .* 2*pi*(freqIW+freqBKG)*this.B0*this.gyro + pini) + ...
            %                 S0EW .* exp(-this.te ./ t2sEW) .* sin(this.te .* 2*pi*(freqEW+freqBKG)*this.B0*this.gyro + pini) .* exp(decayEW) ).*extraData.ff,3);

            if ~fitting.isComplex
                S = sqrt(Sreal.^2 + Simag.^2);
            else
                S = cat(1,Sreal,Simag);
            end
            
        end

        % DIMWI model (compatible to conventional 3-pool model)
        function S = model_DIMWI(this, Amw, Aiw, Aew, t2smw, t2siw, t2sew, decay_ew, freqmw, freqiw, freqew, freqbkg, pini)
            
            S = (Amw .* exp(this.te .* (-1./t2smw + 1i*2*pi*freqmw)) + ...                % S_MW
                 Aiw .* exp(this.te .* (-1./t2siw + 1i*2*pi*freqiw)) + ...                % S_IW
                 Aew .* exp(this.te .* (-1./t2sew + 1i*2*pi*freqew)).*exp(-decay_ew)) .* ...   % S_EW
                 exp(1i*pini) .* exp(1i*2*pi*freqbkg.*obj.te);                          % phase offset and total field
        end

        % Conventonal 3-pool GRE-MWI model
        function S = model_3pool(this, Amw, Aiw, Aew, t2smw, t2siw, t2sew, freqmw, freqiw, freqew, freqbkg, pini)

            % Smw = Amw .* exp(this.te .* (-1./t2smw + 1i*2*pi*freqmw));
            % Siw = Aiw .* exp(this.te .* (-1./t2siw + 1i*2*pi*freqiw));
            % Sew = Aew .* exp(this.te .* (-1./t2sew + 1i*2*pi*freqew));

            S = (Amw .* exp(this.te .* (-1./t2smw + 1i*2*pi*freqmw)) + ...
                 Aiw .* exp(this.te .* (-1./t2siw + 1i*2*pi*freqiw)) + ...
                 Aew .* exp(this.te .* (-1./t2sew + 1i*2*pi*freqew))) .* exp(this.te .* 2*pi*freqbkg + pini);

        end
        
%% Starting point estimation

        % determine how the starting points will be set up
        function x0 = determine_x0(this,y,fitting) 

            % Nv: # voxels
            [~, Nv] = size(y);

            if ischar(fitting.start)
                switch lower(fitting.start)
                    case 'likelihood'
                        % using maximum likelihood method to estimate starting points
                        % x0 = this.estimate_prior(y,fitting.lmax);
    
                    case 'default'
                        % use fixed points
                         fprintf('Using default starting points for all voxels at [%s]: [%s]\n\n',this.cell2str(this.model_params),replace(num2str(this.startpoint.',' %.2f'),' ',','));

                        x0 = repmat(this.startpoint, 1, Nv);
                    
                end
            else
                % user defined starting point
                x0 = fitting.start(:);
                fprintf('Using default starting points for all voxels at [%s]: [%s]\n\n',this.cell2str(this.model_params),replace(num2str(x0.',' %.2f'),' ',','));
                
                x0 = repmat(x0, 1, Nv);
            end

            fprintf('Estimation lower bound [%s]: [%s]\n',      this.cell2str(this.model_params), replace(num2str(fitting.boundary(:,1).',' %.2f'),' ',','));
            fprintf('Estimation upper bound [%s]: [%s]\n\n',    this.cell2str(this.model_params), replace(num2str(fitting.boundary(:,2).',' %.2f'),'  ',','));


        end

        % using maximum likelihood method to estimate starting points
        function pars0 = estimate_prior(this,y,lmax)
            % using maximum likelihood method to estimate starting points
            disp('Estimate starting points based on likelihood ...')
            pool                = gcp('nocreate');
            isDeletepool        = false;
            N_sample            = 1e4;
            [x_train, S_train]  = this.traindata(N_sample,lmax);
            if isempty(pool)
                Nworker         = min(max(8,floor(maxNumCompThreads/4)),maxNumCompThreads);
                pool            = parpool('Processes',Nworker);
                isDeletepool    = true;
            end

            % if lmax == 0
                % Nparam = numel(this.model_params) - 2;
            % elseif lmax == 2
                Nparam = numel(this.model_params) - 1;
            % end

            pars0 = zeros(Nparam,size(y,2));
            start = tic;
            % loop all voxels
            parfor k = 1:size(y,2)
                pars0(:,k) = this.likelihood(y(:,k), x_train, S_train, lmax);
            end
            ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
            fprintf('Starting points estimated. Elapsed time (hh:mm:ss): %s \n',string(ET));
            if isDeletepool
                delete(pool);
            end
            % noise
            pars0(Nparam+1,:) = this.startpoint(end);

        end


%% Utilities

        % normalise input data based on masked signal intensity at 98%
        function [img, scaleFactor] = prepare_data(this,img, mask)

            [~,S0] = this.R2star_trapezoidal(abs(img),this.te);

            scaleFactor = prctile( S0(mask), 98);

            img = img ./ scaleFactor;

        end

        function [extraData] = validate_data(this,data,extraData)

            dims = size(data,1:3);

            if ~isfield(extraData,'ff')
                extraData.ff = ones(dims);
            end
            if ~isfield(extraData,'theta')
                extraData.theta = zeros(dims);
            end
            if ~isfield(extraData,'freqBKG')
                extraData.freqBKG = zeros(dims);
            else
                extraData.freqBKG = extraData.freqBKG / (this.B0*this.gyro);
            end
            if ~isfield(extraData,'pini')
                extraData.pini = zeros(dims);
            end

        end
        
        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default(this,fitting)
            % get basic fitting setting check
            fitting2 = mcmc.check_set_default_basic(fitting);

            % if ~isfield(fitting,'isWeighted')
            %     fitting2.isWeighted = true;
            % end
            % if ~isfield(fitting,'weightPower')
            %     fitting2.weightPower = 2;
            % end

            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitFreqMW')
                fitting2.DIMWI.isFitFreqMW = true;
            end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitFreqIW')
                fitting2.DIMWI.isFitFreqIW = true;
            end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitR2sEW')
                fitting2.DIMWI.isFitR2sEW = true;
            end
            if ~isfield(fitting,'DIMWI') || ~isfield(fitting.DIMWI,'isFitVic')
                fitting2.DIMWI.isFitVic = true;
            end
            % 
            % % update properties given lmax
            % this.updateProperty(fitting);

            % get fitting algorithm setting
            if ~isfield(fitting,'start')
                fitting2.start = 'default';
            end
            if ~isfield(fitting,'boundary')
                % otherwise uses default
                fitting2.boundary   = cat(2, this.lb(:),this.ub(:));
            end
            
        end

    end
    
    methods(Static)

        function [weights, fieldmapSD]= compute_optimum_weighting_combining_phase_difference(magn, TE)

            dims    = size(magn);
            dims(4) = dims(4) - 1;
                
            fieldmapSD = zeros(dims, 'like', magn);   % fieldmap SD
            for k = 1:dims(4)
                fieldmapSD(:,:,:,k) = 1./(TE(k+1)-TE(1)) ...
                    * sqrt((magn(:,:,:,1).^2+magn(:,:,:,k+1).^2)./((magn(:,:,:,1).*magn(:,:,:,k+1)).^2));
            end
            
            % weights are inverse of the field map variance
            weights = bsxfun(@rdivide,1./(fieldmapSD.^2),sum(1./(fieldmapSD.^2),4));
        
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

        function st = cell2str(cellStr)
            cellStr= cellfun(@(x){[x ',']},cellStr);  % Add ',' after each string.
            st = cat(2,cellStr{:});  % Convert to string
            st(end) = [];  % Remove last ','
        end
    end

end     

