classdef mcmc < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% 
% This is the class of all MCMC realted functions
%
% Date created: 13 June 2024 
% Date modified:
    properties (GetAccess = public, SetAccess = protected)

    end

    methods(Static)
        

        function [xExpected,xPosterior] = metropilis_hastings(y,x0,xStepsize,fitting,modelFWD,varargin)

            % convert data into single datatype for better performance
            y   = gpuArray( single(y) );

            % Nm: # measurements
            % Nv: # voxels
            [Nm, Nv] = size(y);

            boundary    = gpuArray(single(fitting.boundary));
            lb          = repmat(boundary(:,1),1,Nv);
            ub          = repmat(boundary(:,2),1,Nv);

            % initialize exponent of old probability
            % compute the #samples in posterior distribution
            Ns          = floor( (fitting.iteration - floor(fitting.iteration*fitting.burnin)) / fitting.sampling ) + 1;
            xPosterior  = zeros(size(x0,1), Nv, Ns, fitting.repetition,'single');
    
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
                % 1. Sample the next parameter combination
                xNew                    = xOld + xStepsize.*randn(size(xOld),'like',xOld);
                isOutofbound            = max(max(xNew<lb,[],1), max(xNew>ub,[],1));    % determine step out of boundary
                % replace boundary values so it does not give error when compting probability
                xNew(xNew<lb)           = lb(xNew<lb);
                xNew(xNew>ub)           = ub(xNew>ub);

                % 2. Metropolis sampling
                % If the probability ratio of new to old > threshold,
                % we take the new solution.
                % 2.1 new probability
                logPNew                 = arrayfun(@logP, sum( (modelFWD(xNew(1:end-1, :), varargin{:})-y).^2, 1 ), xNew(end,:), Nm);
                logPNew(isOutofbound)   = -10^20; % set a very low probability if out of bound

                % 2.2 Compute acceptance ratio based on new/old probability
                acceptanceRatio         = min(exp(logPNew-logPOld), 1);
                isAccepted              = acceptanceRatio > rand(1,Nv,'like',xOld);
                % 2.3 update parameters
                logPOld(isAccepted)     = logPNew(isAccepted);
                xOld(:,isAccepted)      = xNew(:,isAccepted);

                % 3. Maintain the independence between iterations
                % 3.1 discard the first burnin*100% iterations
                % 3.2 keep an iteration every N iterations
                if ( k > floor(fitting.iteration*fitting.burnin)+1 ) && ( mod(k, fitting.sampling)==1 )
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

            % average over distribution
            switch fitting.method
                case 'mean'
                    xExpected = squeeze(mean(mean(xPosterior,3),4));
                case 'median'
                    xExpected = squeeze(median(median(xPosterior,3),4));
            end

            disp('The process is completed.')

            end

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
        %
            fitting2 = fitting;

            % get fitting algorithm setting
            if ~isfield(fitting,'iteration')
                fitting2.iteration = 2e5;
            end
            if ~isfield(fitting,'sampling')
                fitting2.sampling = 100;
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
        end

        % display fitting algorithm parameters
        function display_basic_algorithm_parameters(fitting)
            
            disp('----------------------------------------------------');
            disp('Markov Chain Monte Carlo (MCMC) algorithm parameters');
            disp('----------------------------------------------------');
            disp(['No. of iterations : ', num2str(fitting.iteration)]);
            disp(['No. of repetitions: ', num2str(fitting.repetition)])
            disp(['Sampling interval : ', num2str(fitting.sampling)]);
            disp(['Method            : ', fitting.method ]);

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
                out.posterior.(model_params{k}) = single(squeeze(xPosterior(1,:,:,:))); 
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