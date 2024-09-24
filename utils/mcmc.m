classdef mcmc < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% 
% This is the class of all MCMC related functions
%
% Date created: 13 June 2024 
% Date modified: 7 August 2024
% Date modified: 23 August 2024
%
    properties (GetAccess = public, SetAccess = protected)

    end

    methods
        function out = optimisation(this, data, mask, weights, pars0, fitting, FWDfunc, varargin)
        % Input
        % ----------
        % data          : N-D measurement data, First 3 dims reserve for spatial info
        % mask          : M-D signal mask (M=[1,3])
        % weights       : N-D weights for optimisaiton, same dim as 'data'
        % pars0         : Structure variable containing all parameters to be estimated
        % fitting       : Structure variable containing all fitting algorithm setting
        %   .model_params       : 1xM cell variable,    name of the model parameters, e.g. {'S0','R2star','noise'};
        %   .lb                 : 1xM numeric variable, fitting lower bound, same order as field 'model_params', e.g. [0.5, 0, 0.001];
        %   .ub                 : 1xM numeric variable, fitting upper bound, same order as field 'model_params', e.g. [2, 1, 0.1];
        %   .algorithm          : MCMC algorithm, 'MH'|'GW'
        %   .iteration          : # MCMC iterations
        %   .thinning           : sampling interval between iterations
        %   .burnin             : iterations at the beginning to be discarded, if burnin>1, then the exact number will  be used; if 0<burnin<1 then actual burnin = iteration*burnin
        %   .repetition         : # repetition of MCMC proposal
        %   .xStepSize          : step size of model parameter in MCMC proposal, same size and order as 'model_params' ('MH' only)
        %   .StepSize           : step size for 'GW' in MCMC proposal ('GW' only)
        %   .Nwalker            : # random walkers ('GW' only)
        % FWDfunc       : function handle of forward model
        % varargin      : contains additional input requires for FWDfunc
        % 

            fitting = this.check_set_default_basic(fitting);

            % Step 0: display basic messages
            this.display_basic_algorithm_parameters(fitting);

            % mask data to reduce memory load
            data = this.vectorise_NDto2D(data,mask).';
            if ~isempty(weights); weights = this.vectorise_NDto2D(weights,mask).'; else; weights = ones(size(data),'like',data); end
            for km = 1:numel(fitting.model_params); pars0.(fitting.model_params{km}) = this.vectorise_NDto2D( pars0.(fitting.model_params{km}), mask).' ;end

            % MCMC
            if strcmpi(fitting.algorithm,'mh')
                xPosterior = this.metropolis_hastings(data, pars0, weights, fitting, FWDfunc ,varargin{:});
            else
                xPosterior = this.goodman_weare(data, pars0, weights, fitting, FWDfunc ,varargin{:});
            end

            % finish up
            out = this.res2out(xPosterior,fitting,mask);

        end

        function xPosterior = metropolis_hastings(this,y,x0,weights,fitting,FWDfunc,varargin)
        % Input
        % ------
        % y         : measurements, [Nmeas,Nvoxels]
        % x0        : structure array, starting points, N fields, each field 1xNvoxel
        % weights   : weighting for non-linear least square fitting, same dimension as y
        % fitting       : Structure variable containing all fitting algorithm setting
        %   .model_params       : 1xM cell variable,    name of the model parameters, e.g. {'S0','R2star','noise'};
        %   .lb                 : 1xM numeric variable, fitting lower bound, same order as field 'model_params', e.g. [0.5, 0, 0.001];
        %   .ub                 : 1xM numeric variable, fitting upper bound, same order as field 'model_params', e.g. [2, 1, 0.1];
        %   .iteration          : # MCMC iterations
        %   .thinning           : sampling interval between iterations
        %   .burnin             : iterations at the beginning to be discarded, if burnin>1, then the exact number will  be used; if 0<burnin<1 then actual burnin = iteration*burnin
        %   .repetition         : # repetition of MCMC proposal
        %   .xStepSize          : step size of model parameter in MCMC proposal, same size and order as 'model_params' ('MH' only)
        % FWDfunc   : function handle for forward signal model
        % varargin  : other input required for @FWDfunc
        %

            fitting = this.check_set_default_basic(fitting);
            if isempty(weights); weights = ones(size(y), 'like', y); end

            % Nm: # measurements; Nv: # voxels
            [Nm, Nv]    = size(y);
            % Nvar: # estimation parameters
            Nvar        = numel(fitting.model_params);
            Nburnin     = this.get_number_burnin(fitting);
            % Ns: # samples in posterior distribution
            Ns          = numel(Nburnin+1:fitting.thinning:fitting.iteration);  %floor( (fitting.iteration - floor(fitting.iteration*fitting.burnin)) / fitting.thinning );

            % convert data into single datatype for better performance and out themn into GPU
            y       = gpuArray( single(y) );
            weights = gpuArray( single(weights) );
            for km = 1:Nvar; x0.(fitting.model_params{km}) = gpuArray(single( x0.(fitting.model_params{km}) ));end
            xStepsize = gpuArray(single(fitting.xStepSize(:)));
            % setup boundary variables
            lb          = gpuArray( single(repmat(fitting.lb(:),1,Nv)));
            ub          = gpuArray( single(repmat(fitting.ub(:),1,Nv)));
            % initialize array to staore all the samples
            xPosterior  = zeros(Nvar, Nv, Ns, fitting.repetition,'single');
    
            % compute likelihood at starting points
            % logP is converted into external function for specific CUDA kernel 
            % logP = @(X, Y) -sum( (this.FWD(X(1:4, :), model)-Y).^2, 1 )./(2*X(5,:).^2) + Nm/2*log(1./X(5,:).^2);
            xCurr   = this.struct2array(x0,fitting.model_params);      % extract parameter structure to numeric array for faster computation
            xCurr   = max(xCurr,lb); xCurr = min(xCurr,ub);            % set boundary
            x0      = this.array2struct(xCurr,fitting.model_params);   % convert array back to structure for FWD function
            logP0   = arrayfun(@logP_Gaussian, sum( weights.* (FWDfunc(x0,varargin{:})-y).^2, 1 ), x0.noise, Nm);

            disp('-------------------------');
            disp('MCMC optimisation process');
            disp('-------------------------');

            % loop (multiple) proposal (same start)
            for ii = 1:fitting.repetition
            fprintf('Repetition #%i/%i \n',ii,fitting.repetition)

            % reset start point
            logPCurr    = logP0;
            xCurr       = this.struct2array(x0,fitting.model_params);

            counter = 0; start = tic;
            for k = 1:fitting.iteration
                % 1. make a proposal with normal distribution
                % proposal is generated during iteration
                xProposed       = xCurr + xStepsize.*randn(size(xCurr),'like',xCurr);
                % find proposal that is out of bound for exclusion
                isOutofbound    = max(or(xProposed<lb, xProposed>ub),[],1);    
                % replace out of bound by boundary values to avoid error when compting probability
                xProposed = max(xProposed,lb); xProposed = min(xProposed,ub);
                % convert the proposal into structure array for FWD function
                xProposed_struct = this.array2struct(xProposed,fitting.model_params);

                % 2. Metropolis sampling
                % If the probability ratio of new to old > threshold, we take the new solution.
                % 2.1 proposal probability
                logPProposed            = arrayfun(@logP_Gaussian, sum( weights.* (FWDfunc(xProposed_struct, varargin{:})-y).^2, 1 ), xProposed_struct.noise, Nm);
                % 2.2 Compute acceptance ratio based on new/old probability
                acceptanceRatio         = min(exp(logPProposed-logPCurr), 1);
                isAccepted              = acceptanceRatio > rand(1,Nv,'like',logPProposed);
                isAccepted(isOutofbound)= 0;    % reject out of bound proposal
                % 2.3 update parameters if accepted
                logPCurr(isAccepted)    = logPProposed(isAccepted);
                xCurr(:,isAccepted)     = xProposed(:,isAccepted);

                % 3. Maintain the independence between iterations
                % 3.1 discard the first burnin*100% iterations
                % 3.2 keep an iteration every N iterations
                if ( k > Nburnin ) && mod(k-Nburnin+1, fitting.thinning) == 0 %( mod(k, fitting.thinning)==1 )
                    counter = counter+1;
                    xPosterior(:,:,counter,ii) = gather(xCurr);
                end

                % display message at 1000 iteration and every 10000 iteration
                if mod(k,fitting.iteration/50) == 0 || k == min(1e3, fitting.iteration/100)
                    ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
                    ERT = ET / (k/fitting.iteration) - ET;
                    fprintf('Iteration #%6d,    Elapsed time (hh:mm:ss):%s,     Estimated remaining time (hh:mm:ss):%s \n',k,string(ET),string(ERT));
                end
            end
            end

            % convert final posterior distribution into structure
            xPosterior = this.array2struct(xPosterior,fitting.model_params);
            for kvar = 1:Nvar; xPosterior.(fitting.model_params{kvar}) = shiftdim(xPosterior.(fitting.model_params{kvar}),1); end

            disp('The process is completed.')

        end

        function xPosterior = goodman_weare(this,y,x0,weights,fitting,modelFWD,varargin)
        % Input
        % ----------
        % y         : measurements, [Nmeas,Nvoxels]
        % x0        : structure array, starting points, N fields, each field 1xNvoxel
        % weights   : weighting for non-linear least square fitting, same dimension as y
        % pars0         : Structure variable containing all parameters to be estimated
        % fitting       : Structure variable containing all fitting algorithm setting
        %   .model_params       : 1xM cell variable,    name of the model parameters, e.g. {'S0','R2star','noise'};
        %   .lb                 : 1xM numeric variable, fitting lower bound, same order as field 'model_params', e.g. [0.5, 0, 0.001];
        %   .ub                 : 1xM numeric variable, fitting upper bound, same order as field 'model_params', e.g. [2, 1, 0.1];
        %   .iteration          : # MCMC iterations
        %   .thinning           : sampling interval between iterations
        %   .burnin             : iterations at the beginning to be discarded, if burnin>1, then the exact number will  be used; if 0<burnin<1 then actual burnin = iteration*burnin
        %   .repetition         : # repetition of MCMC proposal
        %   .StepSize           : step size for 'GW' in MCMC proposal ('GW' only)
        %   .Nwalker            : # random walkers ('GW' only)
        % FWDfunc       : function handle of forward model
        % varargin      : contains additional input requires for FWDfunc
        % 
        % ENSEMBLE SAMPLERS WITH AFFINE INVARIANCE (2010) JONATHAN GOODMAN AND JONATHAN WEARE
        % Other references:
        % emcee: The MCMC Hammer (2013) https://arxiv.org/pdf/1202.3665
        % https://github.com/grinsted/gwmcmc/tree/master
        % 

            fitting = this.check_set_default_basic(fitting);
            if isempty(weights); weights = ones(size(y), 'like', y); end

            % Nm: # measurements; Nv: # voxels
            [Nm, Nv] = size(y);
            % Nvar: # estimation parameters
            Nvar     = numel(fitting.model_params);
            Nburnin  = this.get_number_burnin(fitting);
            % Ns: # samples in posterior distribution
            Ns       = numel(Nburnin+1:fitting.thinning:fitting.iteration);
            Nwalker  = fitting.Nwalker;
            StepSize = fitting.StepSize;
        
            % convert data into single datatype for better performance
            y       = gpuArray( single(y) );
            weights = gpuArray( single(weights) );
            for km = 1:Nvar; x0.(fitting.model_params{km}) = gpuArray(single( x0.(fitting.model_params{km}) ));end
        
            % setup boundary variables
            lb          = gpuArray( single(repmat(fitting.lb(:),1,Nv,Nwalker)));
            ub          = gpuArray( single(repmat(fitting.ub(:),1,Nv,Nwalker)));
            % set up weight
            weights     = repmat(weights,1,1,Nwalker);
            % initialize array to staore all the samples
            xPosterior  = zeros(Nvar, Nv, Nwalker, Ns, fitting.repetition,'single');
            
            % initiate an ensemble of walkers around the starting position (0.1% full range) with Gaussian distribution
            % 1st: Nvar;2nd: Nv; 3rd: Nwalker
            xCurr   = this.struct2array(x0,fitting.model_params);       % extract parameter structure to numeric array for faster computation
            xCurr   = xCurr + (ub-lb)*0.001.*randn(size(ub));           % initiate starting position for all walkers
            xCurr   = max(xCurr,lb); xCurr = min(xCurr,ub);             % set boundary
            x0      = this.array2struct(xCurr,fitting.model_params);    % convert array back to structure for FWD function
            % compute likelihood at starting points
            logP0   = arrayfun(@logP_Gaussian, sum( weights.* (modelFWD(x0, varargin{:})-y).^2, 1 ), x0.noise, Nm);
        
            disp('-------------------------');
            disp('MCMC optimisation process');
            disp('-------------------------');
        
            for ii = 1:fitting.repetition
            fprintf('Repetition #%i/%i \n',ii,fitting.repetition)
        
            logPCurr= logP0;
            xCurr   = this.struct2array(x0,fitting.model_params);
        
            counter = 0; start = tic;
            for k = 1:fitting.iteration
                % 1. make a proposal with normal distribution
                % 1.1. find a unique partner for a walker k
                partner         = this.find_partner(Nwalker);
                % 1.2. stretch move
                zz              = ((StepSize-1)*rand(size(logP0),'like',xCurr) + 1).^2 / StepSize;
                xProposed       = xCurr(:,:,partner) + (xCurr - xCurr(:,:,partner)).*zz;
                % find proposal that is out of bound for exclusion
                isOutofbound    = max(or(xProposed<lb, xProposed>ub),[],1);    
                % replace boundary values so it does not give error when compting probability
                xProposed = max(xProposed,lb); xProposed = min(xProposed,ub);
                % convert the proposal into structure array for FWD function
                xProposed_struct = this.array2struct(xProposed,fitting.model_params);
        
                % 2. Metropolis sampling
                % If the probability ratio of new to old > threshold, we take the new solution.
                % 2.1 proposal probability
                logPProposed            = arrayfun(@logP_Gaussian, sum( weights.* (modelFWD(xProposed_struct, varargin{:})-y).^2, 1 ), xProposed_struct.noise, Nm);
                % 2.2 Compute acceptance ratio based on z^(Nd-1)*new/old probability
                acceptanceRatio         = min(zz.^(Nvar-1).*exp(logPProposed-logPCurr), 1);
                isAccepted              = acceptanceRatio > rand(1,Nv,Nwalker,'like',xCurr);
                isAccepted(isOutofbound)= 0;    % reject out of bound proposal
                % 2.3 update parameters
                logPCurr(isAccepted)    = logPProposed(isAccepted);
                xCurr(:,isAccepted)     = xProposed(:,isAccepted);
        
                % 3. Maintain the independence between iterations
                % 3.1 discard the first burnin*100% iterations
                % 3.2 keep an iteration every N iterations
                if ( k > Nburnin ) && mod(k-Nburnin+1, fitting.thinning) == 0 
                    counter = counter+1;
                    xPosterior(:,:,:,counter,ii) = gather(xCurr);
                end
        
                % display message at 1000 ietration and every 2000 iterations
                if mod(k,fitting.iteration/50) == 0 || k == min(1e2, fitting.iteration/100)
                    ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
                    ERT = ET / (k/fitting.iteration) - ET;
                    fprintf('Iteration #%6d,    Elapsed time (hh:mm:ss):%s,     Estimated remaining time (hh:mm:ss):%s \n',k,string(ET),string(ERT));
                end
            end
            end

            % convert final posterior distribution into structure
            xPosterior = this.array2struct(xPosterior,fitting.model_params);
            for kvar = 1:Nvar; xPosterior.(fitting.model_params{kvar}) = shiftdim(xPosterior.(fitting.model_params{kvar}),1); end
        
            disp('The process is completed.')
        
        end

        % convert estimation into organised output structure
        function out = res2out(this,xPosterior,fitting,mask)
            
            % store the unshaped posterior into out
            out.posterior = xPosterior;

            % compute additional metric if specified
            fields = fieldnames(xPosterior);

            Nvox    = size(xPosterior.(fields{1}),1);
            Nsample = prod(size(xPosterior.(fields{1}),2:5));

            metrics = fitting.metric;
            if ~isempty(metrics)
                for km = 1:numel(metrics)
                    switch lower(metrics{km})
                        case 'mean'
                            for kvar=1:numel(fields)
                                tmp = mean( reshape( xPosterior.(fields{kvar}), [Nvox, Nsample]),2);
                                tmp = this.ND2image(tmp,mask);
                                out.mean.(fields{kvar}) = tmp;
                            end
                        case 'median'
                            for kvar=1:numel(fields)
                                tmp = median( reshape( xPosterior.(fields{kvar}), [Nvox, Nsample]),2);
                                tmp = this.ND2image(tmp,mask);
                                out.median.(fields{kvar}) = tmp;
                            end
                        case 'std'
                            for kvar=1:numel(fields)
                                tmp = std( reshape( xPosterior.(fields{kvar}), [Nvox, Nsample]),[],2);
                                tmp = this.ND2image(tmp,mask);
                                out.std.(fields{kvar}) = tmp;
                            end
                        case 'iqr'
                            for kvar=1:numel(fields)
                                tmp = iqr( reshape( xPosterior.(fields{kvar}), [Nvox, Nsample]),2);
                                tmp = this.ND2image(tmp,mask);
                                out.iqr.(fields{kvar}) = tmp;
                            end
                    end
                end
            end


        end

    end

    methods(Static)

        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default_basic(fitting)
        % Input
        % -----
        % fitting       : structure contains fitting algorithm parameters
        %   .iteration  : no. of maximum MCMC iterations,   default = 200k
        %   .repetition : no. of MCMC repetitions,          default = 1
        %   .thinning   : MCMC thinning interval,           default = every 100 iterations
        %   .burning    : MCMC burn-in ratio,               default = 10%
        %   .method     : method to compute expected valur from posterior distribution, 'mean' (default) | 'median'
        %   .algorithm  : MCMC algorithm 'MH': Metropolis-Hastings; 'GW': Goodman-Weare, 'MH' (default) | 'GW' 
        %   .StepSize   : Step size for Goodman-Weare,      default = 2
        %   .Nwalker    : number of walkers for Goodman-Weare,      default = 50
        %
            fitting2 = fitting;

            % get fitting algorithm setting
            if ~isfield(fitting,'iteration')
                fitting2.iteration = 2e5;
            end
            if ~isfield(fitting,'thinning')
                fitting2.thinning = 20;    % thinning, sampled every 100 interval
            end
            if ~isfield(fitting,'metric')
                fitting2.metric = {'mean','std'};
            end
            if ~isfield(fitting,'burnin')
                fitting2.burnin = 0.1;      % 10% burnin
            end
            if ~isfield(fitting,'repetition')
                fitting2.repetition = 1;
            end 
            if ~isfield(fitting,'output_filename')
                fitting2.output_filename = [];
            end
            if ~isfield(fitting,'algorithm')
                fitting2.algorithm = 'MH';
            end
            if ~isfield(fitting,'StepSize')
                fitting2.StepSize = 2;
            end
            if ~isfield(fitting,'Nwalker')
                fitting2.Nwalker = 50;
            end 
            if ~isfield(fitting,'ub')
                fitting2.ub = [];
            end
            if ~isfield(fitting,'lb')
                fitting2.lb = [];
            end
            if ~iscell(fitting2.metric)
                fitting2.metric = cellstr(fitting2.metric);
            end
        end

        % display fitting algorithm parameters
        function display_basic_algorithm_parameters(fitting)

            if strcmpi( fitting.algorithm, 'gw')
                algorithm = 'Affine-Invariant Ensemble';
            else
                algorithm = 'Metropolis-Hastings';
            end

            if fitting.burnin > 1
                burnin = (fitting.burnin ./ fitting.iteration) * 100;
            else
                burnin = fitting.burnin * 100;
            end
            
            disp('----------------------------------------------------');
            disp('Markov Chain Monte Carlo (MCMC) algorithm parameters');
            disp('----------------------------------------------------');
            disp(['Algorithm         : ', algorithm]);
            disp(['No. of iterations : ', num2str(fitting.iteration)]);
            disp(['No. of repetitions: ', num2str(fitting.repetition)])
            disp(['Thinning          : ', num2str(fitting.thinning)]);
            disp(['Burn-in (%)       : '  num2str(burnin)])
            disp(['Metric(s)         : ', cell2str(fitting.metric)]);
            if strcmpi( fitting.algorithm, 'gw'); disp(['Step size         : ', num2str(fitting.StepSize) ]); end
            if strcmpi( fitting.algorithm, 'gw'); disp(['No. of walkers    : ', num2str(fitting.Nwalker) ]); end

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

        % this utility function to convert the MCMC posterior distribution into 4D/5D image
        function img = ND2image(dist,mask)
            
            imageDims = size(mask,1:3);
            extraDims = size(dist,2:ndims(dist));

            % find masked signal
            mask_idx            = find(mask>0);
            % reshape the input to an image         
            img                     = zeros(numel(mask),extraDims,'single'); 
            img(mask_idx,:,:,:,:,:) = dist; 
            img                     = reshape(img, [imageDims, extraDims]);
            
        end

        % initialise parameters
        function parameters = initialise_start(dims,fitting)
            
            % get relevant parameters
            model_params    = fitting.model_params;

            for k = 1:numel(model_params)
                parameters.(model_params{k}) = ones(dims,'single') *fitting.start(k);
            end


        end

        % save the mcmc output structure variable into disk space 
        function save_mcmc_output(output_filename,out)
        % Input
        % ------------------
        % output_filename   : output filename
        % out               : output structure of askadam
        %

            % save the estimation results if the output filename is provided
            if ~isempty(output_filename)
                [output_dir,~,~] = fileparts(output_filename);
                if ~exist(output_dir,'dir')
                    mkdir(output_dir);
                end
                save(output_filename,'out');
                fprintf('Estimation output is saved at %s\n',output_filename);
            end
        end

        % convert numerical array into structure variable for FWD function
        function x_struct = array2struct(x,fields)
            for k = 1:numel(fields)
                x_struct.(fields{k}) = x(k,:,:,:,:,:);
            end
        end

        % convert structure variable into numerical array
        function x = struct2array(x_struct,fields)

            nVol    = size(x_struct.(fields{1}),2);
            nWalker = size(x_struct.(fields{1}),3);

            x = gpuArray(zeros(numel(fields),nVol,nWalker,"single"));
            for k = 1:numel(fields)
                x(k,:,:) = x_struct.(fields{k});
            end
        end

        % compute the number of iteration requires for burn-in
        function Nburnin = get_number_burnin(fitting)
            if fitting.burnin < 1
                Nburnin     = floor(fitting.iteration*fitting.burnin);
            else
                Nburnin     = fitting.burnin;
            end
        end

        % find a unique partner for an index
        function partner = find_partner(maxIndex)
            isSelfPartner = true;
            while isSelfPartner
                partner         = randperm(maxIndex);
                isSelfPartner   = any(partner == 1:maxIndex,'all');
            end
        end

    end
end