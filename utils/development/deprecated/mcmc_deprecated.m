classdef mcmc_deprecated < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% 
% This is the class of all MCMC realted functions
%
% Date created: 13 June 2024 
% Date modified: 7 August 2024
    properties (GetAccess = public, SetAccess = protected)

    end

    methods(Static)
        

        function [xExpected,xPosterior] = metropilis_hastings(y,x0,xStepsize,fitting,modelFWD,varargin)

            % convert data into single datatype for better performance
            y   = gpuArray( single(y) );

            % Nm: # measurements
            % Nv: # voxels
            [Nm, Nv]    = size(y);
            % Ns: # samples in posterior distribution
            Ns          = numel([floor(fitting.iteration*fitting.burnin)+1:fitting.sampling:fitting.iteration]);  %floor( (fitting.iteration - floor(fitting.iteration*fitting.burnin)) / fitting.sampling );
            % Nvar: # estimation parameters
            Nvar        = size(x0,1);

            % setup boundary variables
            boundary    = gpuArray(single(fitting.boundary));
            lb          = repmat(boundary(:,1),1,Nv);
            ub          = repmat(boundary(:,2),1,Nv);

            % initialize exponent of old probability
            xPosterior  = zeros(Nvar, Nv, Ns, fitting.repetition,'single');
    
            % compute likelihood at starting points
            % logP is converted into external function for specific CUDA kernel 
            % logP = @(X, Y) -sum( (this.FWD(X(1:4, :), model)-Y).^2, 1 )./(2*X(5,:).^2) + Nm/2*log(1./X(5,:).^2);
            logP0 = arrayfun(@logP, sum( (modelFWD(x0(1:end-1, :), varargin{:})-y).^2, 1 ), x0(end,:), Nm);

            disp('-------------------------');
            disp('MCMC optimisation process');
            disp('-------------------------');

            for ii = 1:fitting.repetition
            fprintf('Repetition #%i/%i \n',ii,fitting.repetition)

            logPOld = logP0;
            xOld    = x0;

            counter = 0;
            start = tic;
            for k = 1:fitting.iteration
                % 1. make a proposal with normal distribution
                % proposal is generated during iteration
                xNew                    = xOld + xStepsize.*randn(size(xOld),'like',xOld);
                % find proposal that is out of bound for exclusion
                isOutofbound            = max(max(xNew<lb,[],1), max(xNew>ub,[],1));    
                % replace boundary values so it does not give error when compting probability
                xNew(xNew<lb)           = lb(xNew<lb);
                xNew(xNew>ub)           = ub(xNew>ub);

                % 2. Metropolis sampling
                % If the probability ratio of new to old > threshold, we take the new solution.
                % 2.1 proposal probability
                logPNew                 = arrayfun(@logP, sum( (modelFWD(xNew(1:end-1, :), varargin{:})-y).^2, 1 ), xNew(end,:), Nm);
                % logPNew(isOutofbound)   = -10^20; % set a very low probability if out of bound
                % 2.2 Compute acceptance ratio based on new/old probability
                acceptanceRatio         = min(exp(logPNew-logPOld), 1);
                isAccepted              = acceptanceRatio > rand(1,Nv,'like',xOld);
                isAccepted(isOutofbound)= 0;    % reject out of bound proposal
                % 2.3 update parameters if accepted
                logPOld(isAccepted)     = logPNew(isAccepted);
                xOld(:,isAccepted)      = xNew(:,isAccepted);

                % 3. Maintain the independence between iterations
                % 3.1 discard the first burnin*100% iterations
                % 3.2 keep an iteration every N iterations
                if ( k > floor(fitting.iteration*fitting.burnin) ) && mod(k-1, fitting.sampling) == 0 %( mod(k, fitting.sampling)==1 )
                    counter = counter+1;
                    xPosterior(:,:,counter,ii) = gather(xOld);
                end

                % display message at 1000 ietration and every 10000 iteration
                if mod(k,1e4) == 0 || k == 1e3
                    ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
                    ERT = ET / (k/fitting.iteration) - ET;
                    fprintf('Iteration #%6d,    Elapsed time (hh:mm:ss):%s,     Estimated remaining time (hh:mm:ss):%s \n',k,string(ET),string(ERT));
                end
            end
            end

            % average over distribution
            switch fitting.method
                case 'mean'
                    xExpected = squeeze(mean(mean(xPosterior,3),4));
                case 'median'
                    xExpected = squeeze(median(median(xPosterior,3),4)); % TODO: combing the posterior dist. when repetition>1 perhaps?
            end

            disp('The process is completed.')

        end

        function [xExpected,xPosterior] = goodman_weare(y,x0,fitting,modelFWD,varargin)
        % ENSEMBLE SAMPLERS WITH AFFINE INVARIANCE (2010) JONATHAN GOODMAN AND JONATHAN WEARE
        % Other references:
        % emcee: The MCMC Hammer (2013) https://arxiv.org/pdf/1202.3665
        % https://github.com/grinsted/gwmcmc/tree/master
        % 
        
            % convert data into single datatype for better performance
            y   = gpuArray( single(y) );
        
            % Nm: # measurements
            % Nv: # voxels
            [Nm, Nv] = size(y);
            % Ns: # samples in posterior distribution
            Ns          = numel([floor(fitting.iteration*fitting.burnin)+1:fitting.sampling:fitting.iteration]);
            % Nvar: # estimation parameters
            Nvar        = size(x0,1);
        
            Nwalker   = fitting.Nwalker;
            StepSize  = fitting.StepSize;
        
            boundary    = gpuArray(single(fitting.boundary));
            lb          = repmat(boundary(:,1),1,Nv,Nwalker);
            ub          = repmat(boundary(:,2),1,Nv,Nwalker);
        
            % initiate an ensemble of walkers around the starting position (0.1% full range) with Gaussian distribution
            % 1st: Nvar;2nd: Nv; 3rd: Nwalker
            x0  = x0 + (ub-lb)*0.001.*randn(size(ub));
        
            % initialize exponent of old probability
            xPosterior  = zeros(Nvar, Nv, Nwalker, Ns, fitting.repetition,'single');
        
            % compute likelihood at starting points
            % logP is converted into external function for specific CUDA kernel 
            logP0 = arrayfun(@logP, sum( (modelFWD(x0(1:end-1, :,:), varargin{:})-y).^2, 1 ), x0(end,:,:), Nm);
        
            disp('-------------------------');
            disp('MCMC optimisation process');
            disp('-------------------------');
        
            for ii = 1:fitting.repetition
            fprintf('Repetition #%i/%i \n',ii,fitting.repetition)
        
            logPOld = logP0;
            xOld    = x0;
        
            counter = 0;
            start = tic;
            for k = 1:fitting.iteration
                % 1. make a proposal with normal distribution
                % 1.1. find a partner for a walker k
                partner         = randperm(Nwalker);
                % 1.2. stretch move
                zz              = ((StepSize -1)*rand(size(logP0),'like',xOld) + 1).^2 / StepSize;
                xNew            = xOld(:,:,partner) + (xOld - xOld(:,:,partner)).*zz;
                % find proposal that is out of bound for exclusion
                isOutofbound    = max(max(xNew<lb,[],1), max(xNew>ub,[],1));   
                % replace boundary values so it does not give error when compting probability
                xNew(xNew<lb)   = lb(xNew<lb);
                xNew(xNew>ub)   = ub(xNew>ub);
        
                % 2. Metropolis sampling
                % If the probability ratio of new to old > threshold, we take the new solution.
                % 2.1 proposal probability
                logPNew                 = arrayfun(@logP, sum( (modelFWD(xNew(1:end-1, :,:), varargin{:})-y).^2, 1 ), xNew(end,:,:), Nm);
                % 2.2 Compute acceptance ratio based on z^(Nd-1)*new/old probability
                acceptanceRatio         = min(zz.^(Nvar-1).*exp(logPNew-logPOld), 1);
                isAccepted              = acceptanceRatio > rand(1,Nv,Nwalker,'like',xOld);
                isAccepted(isOutofbound)= 0;    % reject out of bound proposal
                % 2.3 update parameters
                logPOld(isAccepted)     = logPNew(isAccepted);
                xOld(:,isAccepted)      = xNew(:,isAccepted);
        
                % 3. Maintain the independence between iterations
                % 3.1 discard the first burnin*100% iterations
                % 3.2 keep an iteration every N iterations
                if ( k > floor(fitting.iteration*fitting.burnin) ) && mod(k-1, fitting.sampling) == 0 
                    counter = counter+1;
                    xPosterior(:,:,:,counter,ii) = gather(xOld);
                end
        
                % display message at 1000 ietration and every 2000 iterations
                if mod(k,2e3) == 0 || k == 1e3
                    ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
                    ERT = ET / (k/fitting.iteration) - ET;
                    fprintf('Iteration #%6d,    Elapsed time (hh:mm:ss):%s,     Estimated remaining time (hh:mm:ss):%s \n',k,string(ET),string(ERT));
                end
            end
            end
        
            % average over distribution
            switch fitting.method
                case 'mean'
                    xExpected = squeeze(mean(mean(reshape(xPosterior,[size(xPosterior,1:2),prod(size(xPosterior,3:4)),size(xPosterior,5)]),3),4));      % concatenate all walkers into a single posterior dist.
                case 'median'
                    xExpected = squeeze(median(median(reshape(xPosterior,[size(xPosterior,1:2),prod(size(xPosterior,3:4)),size(xPosterior,5)]),3),4));  % concatenate all walkers into a single posterior dist.
            end
        
            disp('The process is completed.')
        
        end

        % this utility function to convert the MCMC posterior distribution into 4D/5D image
        function img = distribution2image(dist,mask)
            
            dims = size(mask);

            % find masked signal
            mask_idx        = find(mask>0);
            % reshape the input to an image         
            img                 = zeros(numel(mask),size(dist,2),size(dist,3),'single'); 
            img(mask_idx,:,:)   = dist; 
            img                 = reshape(img,      [dims(1:3), size(dist,2), size(dist,3)]);
            
        end

        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default_basic(fitting)
        % Input
        % -----
        % fitting       : structure contains fitting algorithm parameters
        %   .iteration  : no. of maximum MCMC iterations,   default = 200k
        %   .repetition : no. of MCMC repetitions,          default = 1
        %   .sampling   : MCMC sampling interval,           default = every 100 iterations
        %   .burning    : MCMC burn-in ratio,               default = 10%
        %   .method     : method to compute expected valur from posterior distribution, 'mean' (default) | 'median'
        %   .algorithm  : MCMC algorithm 'MH': Metropolis-Hastings; 'GW': Goodman-Weare, 'MH' (default) | 'GW' 
        %   .StepSize   : Step size for Goodman-Weare,      default = 2
        %
            fitting2 = fitting;

            % get fitting algorithm setting
            if ~isfield(fitting,'iteration')
                fitting2.iteration = 2e5;
            end
            if ~isfield(fitting,'sampling')
                fitting2.sampling = 100;    % thinning, sampled every 100 interval
            end
            if ~isfield(fitting,'method')
                fitting2.method = 'mean';
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
        end

        % display fitting algorithm parameters
        function display_basic_algorithm_parameters(fitting)

            if strcmpi( fitting.algorithm, 'gw')
                algorithm = 'Affine-Invariant Ensemble';
            else
                algorithm = 'Metropolis-Hastings';
            end
            
            disp('----------------------------------------------------');
            disp('Markov Chain Monte Carlo (MCMC) algorithm parameters');
            disp('----------------------------------------------------');
            disp(['Algorithm         : ', algorithm]);
            disp(['No. of iterations : ', num2str(fitting.iteration)]);
            disp(['No. of repetitions: ', num2str(fitting.repetition)])
            disp(['Sampling interval : ', num2str(fitting.sampling)]);
            disp(['Method            : ', fitting.method ]);
            if strcmpi( fitting.algorithm, 'gw'); disp(['Step size         : ', num2str(fitting.StepSize) ]); end
            if strcmpi( fitting.algorithm, 'gw'); disp(['No. of walkers    : ', num2str(fitting.Nwalker) ]); end

        end
        
        % convert estimation into organised output structure
        function out = res2out(xExpected,xPosterior,model_params,mask)

            dims        = size(mask);
            mask_idx    = find(mask>0);

            % organise output structure
            for k = 1:length(model_params)

                % store mean/median into expected value of output structure
                tmp             = zeros(numel(mask),1,'single'); 
                tmp(mask_idx)   = xExpected(k,:); 
                tmp             = reshape(tmp,    dims);
                out.expected.(model_params{k}) = tmp;

                % don't reshape the posterior distributions to avoid memory issue
                out.posterior.(model_params{k}) = single(squeeze(xPosterior(k,:,:,:))); 
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

    end
end