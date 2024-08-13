classdef gpuNEXImcmc < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% AxCaliberSMT model parameter estimation based on MCMC
% Date created: 22 March 2024 
% Date modified: 14 June 2024

    properties
        % default model parameters and estimation boundary
        % Neurite fraction, Neurite axial diffusivity[um2/ms], extracellular radian diffusivity [um2/ms], exchange rate from neurite to extracellular [1/ms], dispersion index,  noise
        model_params    = { 'fa';   'Da';   'De';   'ra';'p2'; 'noise'};
        ub              = [    1;      3;      3;      1;   1;     0.1];
        lb              = [    0;      0;      0;  1/250;   0;    0.01];
        step            = [  0.05;   0.15;   0.15; 0.005;0.05;   0.005];
        startpoint      = [  0.2;      2;    0.5;   0.05; 0.2;    0.05];
        % step            = [ 0.05;   0.15;   0.15;   0.05;0.05;   0.005];
        % startpoint      = [  0.3;      2;    0.5;    0.5; 0.2;    0.05];

    end

    properties (Constant = true, Access = protected)
    end
    
    properties (GetAccess = public, SetAccess = protected)
        b;
        Delta;
        Nav;
        
%         bm2;
    end
    
    properties (GetAccess = private, SetAccess = private)
       
    end
    
    methods (Access = public)
        function this = gpuNEXImcmc(b,  Delta, varargin)
        % gpuAxCaliberSMTmcmc Axon size estimation using AxCaliber-SMT model and MCMC
        % smt = gpumcmcAxCaliberSMT(b, delta, Delta, D0, Da, DeL, Dcsf)
        %       output:
        %           - smt: object of a fitting class
        %
        %       input:
        %           - b: b-value [ms/um2]
        %           - delta: gradient duration [ms]
        %           - Delta: gradient seperation [ms]
        %           - D0: intra-cellular intrinsic diffusivity [um2/ms]
        %           - Da: intra-cellular axial diffusivity [um2/ms]
        %           - DeL: extra-cellular axial diffusivity [um2/ms]
        %           - Dcsf: CSF diffusivity [um2/ms]
        %
        %       usage:
        %           smt = gpumcmcAxCaliberSMT(b, delta, Delta, D0, Da, DeL, Dcsf);
        %           N_sample = 2e6;
        %           model = 'VanGelderen';
        %           N_interval = 100;
        %           [x_median, x] = smt.mcmc(S, N_sample, N_interval, 'median', 0.1);
        %
        %  Authors: 
        %  Kwok-Shing Chan (kchan2@mgh.harvard.edu)
        %  Hong-Hsi Lee (hlee84@mgh.harvard.edu)
        %  Copyright (c) 2022 Massachusetts General Hospital
            
            this.b      = gpuArray( single(b(:)) );
            this.Delta  = gpuArray( single(Delta(:)) );
            if nargin > 2
                this.Nav = varargin{1} ;
            else
                this.Nav =  ones(size(b)) ;
            end
            this.Nav = gpuArray( single(this.Nav(:))) ;
        end

        % update properties according to lmax
        function this = updateProperty(this, fitting)

            if fitting.lmax == 0
                this.model_params(5)    = [];
                this.lb(5)              = [];
                this.ub(5)              = [];
                this.startpoint(5)      = [];
                this.step(5)            = [];
            end

        end

        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('================================================');
            disp('NEXI with Markov Chain Monte Carlo (MCMC) solver');
            disp('================================================');
            
            disp('----------------')
            disp('Data Information');
            disp('----------------')
            fprintf('b-shells (ms/um2)              : [%s] \n',num2str(this.b.',' %.2f'));
            fprintf('Diffusion time (ms)            : [%s] \n\n',num2str(this.Delta.',' %i'));
            fprintf('\n')

        end

        % Perform AxCaliber model parameter estimation based on MCMC across the whole dataset   
        function [out,fa, Da, De, ra, p2, noise] = estimate(this, dwi, mask, extradata, fitting)
        % Input data are expected in multi-dimensional image
        % 
        % Input
        % -----------
        % dwi       : 4D DWI, [x,y,z,dwi]
        % mask      : 3D signal mask, [x,y,z]
        % extradata : extra DWI protocol parameters
        %   .bval   : 1D bval in ms/um2, [1,dwi]
        %   .bvec   : 2D b-table, [3,dwi]
        %   .ldelta : 1D gradient pulse duration in ms, [1,dwi]
        %   .BDELTA : 1D diffusion time in ms, [1,dwi]
        % fitting   : fitting algorithm parameters
        %   .iteration  : number of MCMC iterations
        %   .interval   : interval of MCMC sampling
        %   .method     : method to compute the parameters, 'mean' (default)|'median'
        % 
        % Output
        % -----------
        % out       : output structure contains all MCMC results
        % a         : Axon diameter
        % f         : Neurite volume fraction
        % fcsf      : CSF volume fraction
        % DeR       : radial diffusivity of extracellular water
        % noise     : noise level
        % 
            
            % display basic info
            this.display_data_model_info;
            
            % get all fitting algorithm parameters 
            fitting = this.check_set_default(fitting);

            % compute rotationally invariant signal if needed
            dwi = this.prepare_dwi_data(dwi,extradata,fitting.lmax);

            % vectorise data, 1st dim: b-value*lmax; 2nd dim: voxels
            dwi = DWIutility.vectorise_4Dto2D(dwi,mask).';
            
            % MCMC main
            [x_m, x_dist] = this.fit(dwi, fitting);

            % export results to organise output structure
            out = mcmc.res2out(x_m,x_dist,this.model_params,mask);
            % also export them as variables
            p2 = [];    % in case lmax = 0
            for k = 1:length(this.model_params); eval([this.model_params{k} ' = out.expected.' this.model_params{k} ';']); end
            
            % save the estimation results if the output filename is provided
            mcmc.save_mcmc_output(fitting.output_filename,out)

        end
        
        % Perform parameter estimation using MCMC solver
        function [xExpected,xPosterior] = fit(this, y, fitting)
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
            [xExpected,xPosterior] = mcmc.metropilis_hastings(y,x0,xStepsize,fitting,@this.FWD,fitting);
            
        end

%% Signal generation
        
        % FWD signal model
        function s = FWD(this, pars, fitting)
            fa  = pars(1,:);
            Da  = pars(2,:);
            De  = pars(3,:);
            ra  = pars(4,:);
            
            % Forward model
            if fitting.lmax == 2
                p2 = pars(5,:);
            else
                p2 = [];
            end
            s = this.Sl0Sl2(fa, Da, De, ra, p2);

            % % 1. Sl0
            % s = this.Sl0(fa, Da, De, ra);
            % 
            % % s = arrayfun(@NEXI_Sl0,this.b,this.Delta, fa, Da, De, ra);
            % 
            % if fitting.lmax == 2
            %     p2 = pars(5,:);
            %     s = cat(1,s,this.Sl2(fa, Da, De, ra, p2));
            % end

        end

        function S = Sl0Sl2(this, fa, Da, De, ra, p2)

            bval = this.b;
            DELTA   = this.Delta;
            
            Da = bval.*Da;
            De = bval.*De;
            ra = DELTA.*ra;
            re = ra.*fa./(1-fa);

            % Trapezoidal's rule replacement
            Nx  = 14;    % NRMSE<0.05% for Nx=14
            x   = zeros([ones(1,ndims(fa)), Nx],'like',bval); x(:) = linspace(0,1,Nx); dx = x(2) - x(1);

            M = arrayfun(@NEXI_M,x,fa,Da,De,ra,re);

            % Sl0
            % bypass Matlab's trapz for speed
            S = sum((M(:,:,2:end) + M(:,:,1:end-1)) * (dx) / 2, ndims(x));

            if ~isempty(p2)
                % M = M.*(3*x.^2-1)/2; 
                M = arrayfun(@NEXI_MSl2,M,x);
                % bypass Matlab's trapz for speed
                Sl2 = sum((M(:,:,2:end) + M(:,:,1:end-1)) * (dx) / 2, ndims(x));
                Sl2 = p2.*abs(Sl2);

                S = cat(1,S,Sl2);
            end

        end

        % 0th order rotational invariant
        function S = Sl0(this, fa, Da, De, ra)

            if isgpuarray(fa)
                bval    = gpuArray(single(this.b));
                DELTA   = gpuArray(single(this.Delta));
            else
                bval = this.b;
                DELTA   = this.Delta;
            end

            Da = bval.*Da;
            De = bval.*De;
            ra = DELTA.*ra;
            re = ra.*fa./(1-fa);
            
            % Trapezoidal's rule replacement
            Nx  = 14;    % NRMSE<0.05% for Nx=14
            x   = zeros([ones(1,ndims(fa)), Nx],'like',bval); x(:) = linspace(0,1,Nx); dx = x(2) - x(1);

            % S   = trapz(x(:),this.M(x, fa, Da, De, ra, re),ndims(x));
            % S   = trapz(x(:), arrayfun(@NEXI_M,x,fa,Da,De,ra,re),ndims(x));
            M = arrayfun(@NEXI_M,x,fa,Da,De,ra,re);
            % bypass Matlab's trapz for speed
            S = sum((M(:,:,2:end) + M(:,:,1:end-1)) * (dx) / 2, ndims(x));

        end
        
        % 2nd order rotational invariant
        function S = Sl2(this, fa, Da, De, ra, p2)

            if isgpuarray(fa)
                bval    = gpuArray(single(this.b));
                DELTA   = gpuArray(single(this.Delta));
            else
                bval = this.b;
                DELTA   = this.Delta;
            end

            Da = bval.*Da;
            De = bval.*De;
            ra = DELTA.*ra;
            re = ra.*fa./(1-fa);
            
            % Trapezoidal's rule replacement
            Nx  = 14;    % NRMSE<0.5% for Nx=14
            x   = zeros([ones(1,ndims(fa)), Nx],'like',bval); x(:) = linspace(0,1,Nx); dx = x(2) - x(1);
            % S   = trapz(x(:),this.M(x, fa, Da, De, ra, re).*(3*x.^2-1)/2,ndims(x));
            % S   = trapz(x(:), arrayfun(@NEXI_M,x,fa,Da,De,ra,re).*(3*x.^2-1)/2,ndims(x));
            M = arrayfun(@NEXI_M,x,fa,Da,De,ra,re).*(3*x.^2-1)/2; 
            % bypass Matlab's trapz for speed
            S = sum((M(:,:,2:end) + M(:,:,1:end-1)) * (dx) / 2, ndims(x));

            S   = p2.*abs(S);

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
                        x0 = this.estimate_prior(y,fitting.lmax);
    
                    case 'default'
                        % use fixed points
                        if fitting.lmax == 2
                            fprintf('Using default starting points for all voxels at [fa,Da,De,ra,p2,noise]: [%s]\n\n',replace(num2str(this.startpoint.',' %.2f'),' ',','));
                        else
                            fprintf('Using default starting points for all voxels at [fa,Da,De,ra,noise]: [%s]\n\n',replace(num2str(this.startpoint.',' %.2f'),' ',','));
                        end
                        x0 = repmat(this.startpoint, 1, Nv);
                    
                end
            else
                % user defined starting point
                x0 = fitting.start(:);
                if fitting.lmax == 2
                    fprintf('Using user-defined starting points for all voxels at [fa,Da,De,ra,p2,noise]: [%s]\n\n',replace(num2str(x0.',' %.2f'),' ',','));
                else
                    fprintf('Using user-defined starting points for all voxels at [fa,Da,De,ra,noise]: [%s]\n\n',replace(num2str(x0.',' %.2f'),' ',','));
                end
                
                x0 = repmat(x0, 1, Nv);
            end
            if fitting.lmax == 2
                fprintf('Estimation lower bound [fa,Da,De,ra,p2,noise]: [%s]\n',      replace(num2str(fitting.boundary(:,1).',' %.2f'),' ',','));
                fprintf('Estimation upper bound [fa,Da,De,ra,p2,noise]: [%s]\n\n',    replace(num2str(fitting.boundary(:,2).',' %.2f'),'  ',','));
            else
                fprintf('Estimation lower bound [fa,Da,De,ra,noise]: [%s]\n',      replace(num2str(fitting.boundary(:,1).',' %.2f'),' ',','));
                fprintf('Estimation upper bound [fa,Da,De,ra,noise]: [%s]\n\n',    replace(num2str(fitting.boundary(:,2).',' %.2f'),'  ',','));
            end

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

        % create training data for likelihood
        function [x_train, S_train, intervals] = traindata(this, N_samples, lmax, varargin)
            if nargin < 4
                intervals = [0.01 0.99  ;   % fa
                              1.5 3     ;   % Da
                              0.5 1.5   ;   % De
                                1 100   ;   % exchange time = (1-fa)/r
                             0.01 0.99 ];   % p2
            else
                intervals = varargin{1};
            end
            
            if lmax == 0
                numBSample = numel(this.b);
                numParam   = size(intervals,1) - 1;
            elseif lmax == 2
                numBSample = numel(this.b)*2;
                numParam   = size(intervals,1);
            end
            
            % batch size can be modified according to available hardware
            batch_size  = 1e3;
            reps        = ceil(N_samples/batch_size);
            x_train     = zeros(numParam,batch_size,reps);
            S_train     = zeros(numBSample,batch_size,reps);
            for k = 1:reps
                % generate random parameter guesses and construct batch for NN signal evaluation
                pars = intervals(:,1) + diff(intervals,[],2).*rand(size(intervals,1),batch_size);
                % pars(3,:) = pars(2,:).*pars(3,:);
                pars(4,:) = 1./pars(4,:).*(1-pars(1,:));

                % NEXI Krger signal evaluation
                Sl0 = zeros(numel(this.b),batch_size);
                for j = 1:batch_size
                    Sl0(:,j) = this.Sl0(pars(1,j), pars(2,j), pars(3,j), pars(4,j));
                end

                % in case of Sl2
                if lmax == 2
                    Sl2 = zeros(numel(this.b),batch_size);
                    for j = 1:batch_size
                        Sl2(:,j) = this.Sl2(pars(1,j), pars(2,j), pars(3,j), pars(4,j), pars(5,j)) ;
                    end

                else
                    pars(5,:)   = [];
                    Sl2         = [];
                end

                % remaining signals (dot, soma)
                x_train(:,:,k) = pars;
                S_train(:,:,k) = cat(1,Sl0,Sl2);

            end
            % intervals(3,:) = intervals(2,:).*intervals(3,:);
            intervals(4,:) = (1-intervals(1,end:-1:1))./intervals(4,end:-1:1);
            if lmax == 2
                intervals(5,:) = [];
            end
        end
    
        % likelihood
        function [pars_best, sse_best] = likelihood(this, S0, x_train, S_train,lmax)
            wt = kron(this.Nav(:), 1./(2*(0:2:lmax)+1));
            wt = wt(:);
            nL = floor(lmax/2);
            S0 = S0(1:numel(this.b)*(nL+1),:);
            % batch size can be modified according to available hardware
            [Nx, ~, reps] = size(x_train);
            [~, Nv] = size(S0);
            pars_best = zeros(Nx,Nv);
            sse_best  = inf(1, Nv);
            for k = 1:reps
                pars = x_train(:,:,k);
                S    = S_train(:,:,k);
                for i = 1:Nv
                    S0i = S0(:,i);

                    % scale generated signals (fit S0) to input signal
                    sse = sum(wt.*(S0i - (S0i'*S)./dot(S,S).*S).^2);

                    % store best encountered parameter combination
                    [sse_new,best_index] = min(sse);
                    if sse_new<sse_best(i)
                        sse_best(i)    = sse_new;
                        pars_best(:,i) = pars(:,best_index);
                    end
                end
            end
        end

%% Utilities

        % compute rotationally invariant DWI signal if necessary
        function dwi = prepare_dwi_data(this,dwi,extradata,lmax)

            % full DWI data then compute rotaionally invariant signal
            if size(dwi,4)/(lmax/2+1) > numel(this.b) 
                % compute spherical mean signal
                fprintf('Computing rotationally invariant signal...')

                % if the inout little delta is one value then create a vector
                if numel(extradata.ldelta) == 1
                    extradata.ldelta = ones(size(extradata.bval)) * extradata.ldelta;
                end

                obj = DWIutility();

                [dwi]   = obj.get_Sl_all(dwi,extradata.bval,extradata.bvec,extradata.ldelta,extradata.BDELTA,lmax);

                fprintf('done.\n');

            elseif size(dwi,4) < numel(this.b)
                error('There are more b-shells in the class object than available in the input data. Please check your input data.');
            end

        end
        
        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default(this,fitting)
            % get basic fitting setting check
            fitting2 = mcmc.check_set_default_basic(fitting);

            if ~isfield(fitting,'lmax')
                fitting2.lmax = 0;
            end

            % update properties given lmax
            this.updateProperty(fitting);

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

        %% NEXI signal related
        function M = M(x, fa, Da, d2, r1, r2)
            d1 = Da.*x.^2;
            l1 = (r1+r2+d1+d2)/2;
            l2 = sqrt( (r1-r2+d1-d2).^2 + 4*r1.*r2 )/2; 
            lm = l1-l2;
            Pp = (fa.*d1 + (1-fa).*d2 - lm)./(l2*2);
            M  = Pp.*exp(-(l1+l2)) + (1-Pp).*exp(-lm); 
        end

    end

end     

