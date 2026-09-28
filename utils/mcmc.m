classdef mcmc < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% 
% This is the class of all MCMC related functions
%
% Date created: 13 June 2024 
% Date modified: 7 August 2024
% Date modified: 23 August 2024
% Date modified: 5 October 2024
% Date modified: 4 June 2026 (update affine-invariant method with and without global parameters)
% Date modified: 28 September 2026 (opt-in sampler options for 'MH', moved from mcmc_bayes: parameterTransform,
%                                   updateScheme, adaptStepSize, adaptCovariance, overdisp, R-hat/ESS diagnostics,
%                                   forward-model size check; see metropolis_hastings_adaptive)
%
% Opt-in sampler options ('MH' only)
% -----------------------------------
% With none of these fields set (or all at the defaults below), optimisation runs the
% legacy metropolis_hastings unchanged and the output has no new field. Setting any of them
% runs metropolis_hastings_adaptive instead (same target: Gaussian likelihood, uniform box
% prior on [lb,ub] in native space), and the output gets out.diagnostics and out.settings.
%   .parameterTransform : 'linear'      'linear'|'sigmoid'|'log', or a cell with one entry per modelParams.
%                                       Sampling in u = T(x); the log-target is loglik(x(u)) + sum_p log|dx_p/du_p|
%                                       'linear'  : x = u, box enforced by rejection (as legacy)
%                                       'sigmoid' : x = lb + (ub-lb)*sigmoid(u), needs finite lb < ub, no rejection
%                                       'log'     : x = exp(u), needs finite 0 <= lb < ub, box enforced by rejection
%                                       Start points are clamped to [lb+eps, ub-eps], eps = 1e-4*(ub-lb), before T.
%                                       The initial u-space step is xStepSize ./ |dx/du| at the start point
%                                       (capped at the u-width of the box)
%   .updateScheme       : 'joint'       'joint' (all parameters of a voxel together, 1 forward evaluation per
%                                       iteration) | 'componentwise' (one parameter at a time, Nvar evaluations)
%   .adaptStepSize      : false         Robbins-Monro adaptation of the step size during burn-in, frozen after:
%                                       every adaptInterval iterations, log sigma += 2*j^(-0.6)*(acc_j - adaptTarget),
%                                       acc_j the acceptance rate of the j-th window (per voxel; 'joint': one scale
%                                       for all parameters of a voxel, 'componentwise': one per parameter)
%   .adaptInterval      : 50            # iterations between two adaptation steps
%   .adaptTarget        : []            target acceptance rate, [] -> 0.234 (joint) | 0.44 (componentwise)
%   .adaptCovariance    : false         adaptive Metropolis (Haario et al. 2001), 'joint' + adaptStepSize only:
%                                       per voxel, the running covariance C_i of u (Welford, from iteration
%                                       2*adaptInterval+1) replaces the diagonal proposal at the first adaptation
%                                       step with >= 10*Nvar states, u' = u + lambda_i chol(C_i)*randn with
%                                       lambda_i = 2.38/sqrt(Nvar) then adapted as above; C_i refreshed at every
%                                       adaptation step (previous factor kept if chol fails or < Nvar+1 accepted
%                                       moves); lambda_i and C_i frozen after burn-in
%   .overdisp           : 0             over-dispersed start of repetition ii > 1: u0 + overdisp*W.*randn,
%                                       W = T(ub)-T(lb), clamped to the box (0: all repetitions start at pars0)
% Output with these options: out.diagnostics.acceptance [x,y,z,Nblock,Nrep] (after burn-in), .stepSize.(param)
%   (final u-space step), .ess.(param) (multi-chain bulk ESS), .rhat.(param) (split-R-hat, repetition > 1;
%   both NaN with fewer than 4 samples per chain), .adaptCovariance (adaptCovariance only); out.settings holds
%   the resolved options and the RNG states. See mcmc_bayes for the full derivations.
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
        %   .modelParams       : 1xM cell variable,    name of the model parameters, e.g. {'S0','R2star','noise'};
        %   .lb                 : 1xM numeric variable, fitting lower bound, same order as field 'modelParams', e.g. [0.5, 0, 0.001];
        %   .ub                 : 1xM numeric variable, fitting upper bound, same order as field 'modelParams', e.g. [2, 1, 0.1];
        %   .algorithm          : MCMC algorithm, 'MH'|'GW'
        %   .iteration          : # MCMC iterations
        %   .thinning           : sampling interval between iterations
        %   .burnin             : iterations at the beginning to be discarded, if burnin>1, then the exact number will  be used; if 0<burnin<1 then actual burnin = iteration*burnin
        %   .repetition         : # repetition of MCMC proposal
        %   .xStepSize          : step size of model parameter in MCMC proposal, same size and order as 'modelParams' ('MH' only)
        %   .StepSize           : step size for 'GW' in MCMC proposal ('GW' only)
        %   .Nwalker            : # random walkers ('GW' only)
        %   (opt-in, 'MH' only) .parameterTransform, .updateScheme, .adaptStepSize, .adaptInterval,
        %                       .adaptTarget, .adaptCovariance, .overdisp, see the class header
        % FWDfunc       : function handle of forward model
        % varargin      : contains additional input requires for FWDfunc
        %

            fitting = this.check_set_default_basic(fitting);

            % opt-in sampler options: any of them set -> metropolis_hastings_adaptive (MH only)
            isInfra = mcmc.use_sampler_infra(fitting);
            if isInfra
                if ~strcmpi(fitting.algorithm,'mh')
                    error('mcmc:unsupportedAlgorithm', ...
                        'mcmc: fitting.parameterTransform/updateScheme/adaptStepSize/adaptCovariance/overdisp need fitting.algorithm = ''MH'' (got ''%s'').', fitting.algorithm);
                end
                fitting = this.check_set_default_infra(fitting);
            end

            % Step 0: display basic messages
            this.display_basic_algorithm_parameters(fitting);
            if isInfra; this.display_infra_algorithm_parameters(fitting); end

            % mask data to reduce memory load
            mask_idx = find(mask>0);
            if ~ismatrix(data);     data    = utils.reshape_ND2GD(data,      mask_idx); else; data = data(:,mask_idx);     end
            if ~ismatrix(weights);  weights = utils.reshape_ND2GD(weights,   mask_idx); elseif ~isempty(weights); weights = weights(:,mask_idx);  end
            % data = utils.reshape_ND2GD(data,mask);
            % if ~isempty(weights); weights = utils.reshape_ND2GD(weights,mask); else; weights = ones(size(data), 'like', data); end
            pars0 = utils.reshape_ND2GD_struct(pars0,mask);

            isGlobal = false;
            for k = 1:numel(fitting.modelParams)
                if k == 1
                    N = numel(pars0.(fitting.modelParams{k}));
                else
                    if N ~= numel(pars0.(fitting.modelParams{k})) && numel(pars0.(fitting.modelParams{k})) == 1
                        isGlobal = true;
                        break;
                    end
                end
            end

            % MCMC
            if strcmpi(fitting.algorithm,'mh') && isInfra
                [xPosterior, diagnostics] = this.metropolis_hastings_adaptive(data, pars0, weights, fitting, FWDfunc ,varargin{:});
            elseif strcmpi(fitting.algorithm,'mh')
                xPosterior = this.metropolis_hastings(data, pars0, weights, fitting, FWDfunc ,varargin{:});
            else
                if ~isGlobal
                    xPosterior = this.goodman_weare(data, pars0, weights, fitting, FWDfunc ,varargin{:});
                else
                    xPosterior = this.goodman_weare_wglobal_constant(data, pars0, weights, fitting, FWDfunc ,varargin{:});
                end
            end

            % finish up
            out = this.res2out(xPosterior,fitting,mask);
            if isInfra
                out             = mcmc.diagnostics2out(out, xPosterior, diagnostics, mask);
                out.settings    = diagnostics.settings;
            end

        end

        function xPosterior = metropolis_hastings(this,y,x0,weights,fitting,FWDfunc,varargin)
        % Input
        % ------
        % y         : measurements, [Nmeas,Nvoxels]
        % x0        : structure array, starting points, N fields, each field 1xNvoxel
        % weights   : weighting for non-linear least square fitting, same dimension as y
        % fitting       : Structure variable containing all fitting algorithm setting
        %   .modelParams       : 1xM cell variable,    name of the model parameters, e.g. {'S0','R2star','noise'};
        %   .lb                 : 1xM numeric variable, fitting lower bound, same order as field 'modelParams', e.g. [0.5, 0, 0.001];
        %   .ub                 : 1xM numeric variable, fitting upper bound, same order as field 'modelParams', e.g. [2, 1, 0.1];
        %   .iteration          : # MCMC iterations
        %   .thinning           : sampling interval between iterations
        %   .burnin             : iterations at the beginning to be discarded, if burnin>1, then the exact number will  be used; if 0<burnin<1 then actual burnin = iteration*burnin
        %   .repetition         : # repetition of MCMC proposal
        %   .xStepSize          : step size of model parameter in MCMC proposal, same size and order as 'modelParams' ('MH' only)
        % FWDfunc   : function handle for forward signal model
        % varargin  : other input required for @FWDfunc
        %

            fitting = this.check_set_default_basic(fitting);
            if isempty(weights); weights = ones(size(y), 'like', y); end

            % Nm: # measurements; Nv: # voxels
            [Nm, Nv]    = size(y);
            % Nvar: # estimation parameters
            Nvar        = numel(fitting.modelParams);
            Nburnin     = this.get_number_burnin(fitting);
            % Ns: # samples in posterior distribution
            Ns          = numel(Nburnin+1:fitting.thinning:fitting.iteration);  %floor( (fitting.iteration - floor(fitting.iteration*fitting.burnin)) / fitting.thinning );

            % convert data into single datatype for better performance and out themn into GPU
            y       = gpuArray( single(y) );
            weights = gpuArray( single(weights) );
            for km = 1:Nvar; x0.(fitting.modelParams{km}) = gpuArray(single( x0.(fitting.modelParams{km}) ));end
            xStepsize = gpuArray(single(fitting.xStepSize(:)));
            % setup boundary variables
            lb          = gpuArray( single(repmat(fitting.lb(:),1,Nv)));
            ub          = gpuArray( single(repmat(fitting.ub(:),1,Nv)));
            % initialize array to staore all the samples
            xPosterior  = zeros(Nvar, Nv, Ns, fitting.repetition,'single');
    
            % compute likelihood at starting points
            % logP is converted into external function for specific CUDA kernel 
            % logP = @(X, Y) -sum( (this.FWD(X(1:4, :), model)-Y).^2, 1 )./(2*X(5,:).^2) + Nm/2*log(1./X(5,:).^2);
            xCurr   = this.struct2array(x0,fitting.modelParams);      % extract parameter structure to numeric array for faster computation
            xCurr   = max(xCurr,lb); xCurr = min(xCurr,ub);            % set boundary
            x0      = this.array2struct(xCurr,fitting.modelParams);   % convert array back to structure for FWD function
            logP0   = arrayfun(@logP_Gaussian, sum( weights.* (FWDfunc(x0,varargin{:})-y).^2, 1 ), x0.noise, Nm);

            disp('-------------------------');
            disp('MCMC optimisation process');
            disp('-------------------------');

            % loop (multiple) proposal (same start)
            for ii = 1:fitting.repetition
            fprintf('Repetition #%i/%i \n',ii,fitting.repetition)

            % reset start point
            logPCurr    = logP0;
            xCurr       = this.struct2array(x0,fitting.modelParams);

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
                xProposed_struct = this.array2struct(xProposed,fitting.modelParams);

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
            xPosterior = this.array2struct(xPosterior,fitting.modelParams);
            for kvar = 1:Nvar; xPosterior.(fitting.modelParams{kvar}) = shiftdim(xPosterior.(fitting.modelParams{kvar}),1); end

            disp('The Metroplis-Hastings MCMC sampling is completed.')

        end

        function [xPosterior, diagnostics] = metropolis_hastings_adaptive(this,y,x0,weights,fitting,FWDfunc,varargin)
        % Metropolis-Hastings with the opt-in sampler options (class header): parameter
        % transforms, update scheme, step-size/covariance adaptation during burn-in and
        % over-dispersed starts. Same target as metropolis_hastings. This is the sampling loop
        % of mcmc_bayes.metropolis_hastings_bayes restricted to the Gaussian likelihood without
        % priors, with the same random stream, so both give the same chain for the same seed.
        %
        % Input
        % ------
        % Same as metropolis_hastings, plus the opt-in fitting options
        %
        % Output
        % ------
        % xPosterior    : structure, native-space posterior samples, each field [Nvoxel, Nsample, Nrepetition]
        % diagnostics   : structure
        %   .acceptance     : post-burn-in acceptance rate, [Nvoxel, Nblock, Nrepetition], Nblock = 1 (joint) or Nvar (componentwise)
        %   .stepSize       : final u-space proposal scale, [Nvoxel, Nvar, Nrepetition]
        %   .adaptCovariance: final proposal covariance etc. (adaptCovariance only)
        %   .settings       : resolved sampler settings and RNG states
        %
            fitting = this.check_set_default_infra(this.check_set_default_basic(fitting));
            if isempty(weights); weights = ones(size(y), 'like', y); end
            if ~any(strcmp(fitting.modelParams, 'noise'))
                error('mcmc:noNoise', 'mcmc: the Gaussian likelihood requires ''noise'' in fitting.modelParams.');
            end

            % record RNG states before any random number is drawn
            rngState    = rng;
            gpuRngState = parallel.gpu.rng;

            % Nm: # measurements; Nv: # voxels
            [Nm, Nv]    = size(y);
            % Nvar: # estimation parameters
            Nvar        = numel(fitting.modelParams);
            Nburnin     = this.get_number_burnin(fitting);
            % Ns: # samples in posterior distribution
            Ns          = numel(Nburnin+1:fitting.thinning:fitting.iteration);

            % transforms
            method      = this.parse_transform(fitting.parameterTransform, Nvar);
            this.check_transform_bounds(method, fitting.lb, fitting.ub, fitting.modelParams);
            isLinear    = strcmp(method,'linear');
            hasJac      = ~all(isLinear);                           % false -> x == u, skip all Jacobian terms
            isRejectH   = ~strcmp(method,'sigmoid');                % rows whose box is enforced by rejection
            isRejectRow = gpuArray(isRejectH(:));
            code        = gpuArray(single(this.transform_code(method)));  % [Nvar,1] for the fused GPU kernel
            allReject   = all(isRejectH);                           % true -> same bound check as legacy

            % update scheme and adaptation
            isComponent = strcmpi(fitting.updateScheme,'componentwise');
            if isComponent; Nblock = Nvar; else; Nblock = 1; end
            isAdapt     = logical(fitting.adaptStepSize);
            Nadapt      = floor(Nburnin/fitting.adaptInterval);     % # adaptation windows, all within burn-in
            if isAdapt && Nburnin < 2*fitting.adaptInterval
                warning('mcmc:shortBurnin', ...
                    'adaptStepSize is on but Nburnin (%d) < 2*adaptInterval (%d); %d adaptation step(s) only.', ...
                    Nburnin, 2*fitting.adaptInterval, Nadapt);
            end
            % adaptive covariance (joint + adaptStepSize, validated in check_set_default_infra)
            isACov      = logical(fitting.adaptCovariance);
            if isACov
                acWarm  = 2*fitting.adaptInterval;              % warm-up iterations not accumulated
                acMinN  = 10*Nvar;                              % accumulated states needed for the switch
                acScale = 2.38/sqrt(Nvar);                      % initial lambda at the switch
                acEps   = 1e-6;                                 % relative diagonal loading
                acTiny  = 1e-12;                                % ridge, relative to max_p C_pp
                kAd     = (1:Nadapt) .* fitting.adaptInterval;  % adaptation steps
                kSwitch = kAd(find(kAd - acWarm >= acMinN, 1));
                if isempty(kSwitch)
                    kSwitch = NaN;
                    warning('mcmc:adaptCovarianceNoSwitch', ...
                        ['adaptCovariance is on but the burn-in (%d) is too short to accumulate %d states after the ' ...
                         'warm-up of %d iterations at an adaptation step; the diagonal proposal is used throughout.'], ...
                        Nburnin, acMinN, acWarm);
                end
            end

            % convert data into single datatype for better performance and put them into GPU
            y       = gpuArray( single(y) );
            weights = gpuArray( single(weights) );
            for km = 1:Nvar; x0.(fitting.modelParams{km}) = gpuArray(single( x0.(fitting.modelParams{km}) ));end
            xStepsize = gpuArray(single(fitting.xStepSize(:)));
            % setup boundary variables, [Nvar,1], broadcast over voxels
            lb      = gpuArray( single(fitting.lb(:)));
            ub      = gpuArray( single(fitting.ub(:)));
            % u-space bounds of the eps-clamped box (for overdisp and step-size cap)
            uLo     = this.transform_forward(lb, method, lb, ub);
            uHi     = this.transform_forward(ub, method, lb, ub);
            uWidth  = uHi - uLo;
            % initialize array to store all the samples
            xPosterior  = zeros(Nvar, Nv, Ns, fitting.repetition,'single');
            acceptance  = zeros(Nv, Nblock, fitting.repetition,'single');
            stepSize    = zeros(Nv, Nvar, fitting.repetition,'single');
            if isACov
                acPropCov   = zeros(Nv, Nvar, Nvar, fitting.repetition, 'single');
                acLambda    = zeros(Nv, fitting.repetition, 'single');
                acCond      = zeros(Nv, fitting.repetition, 'single');
                acValidOut  = false(Nv, fitting.repetition);
            end

            % log-likelihood of a native-space parameter array [Nvar,Nv]
            loglik  = @(x) mcmc.loglik_gaussian(this.array2struct(x,fitting.modelParams), y, weights, Nm, FWDfunc, varargin{:});

            % starting point in native space (same as metropolis_hastings), then in u space
            xStart  = this.struct2array(x0,fitting.modelParams);      % extract parameter structure to numeric array for faster computation
            xStart  = max(xStart,lb); xStart = min(xStart,ub);         % set boundary
            uStart  = this.transform_forward(xStart, method, lb, ub);

            % the forward model must return one row per measurement and one column per voxel
            this.check_forward_size(FWDfunc(this.array2struct(xStart, fitting.modelParams), varargin{:}), Nm, Nv);

            disp('-------------------------');
            disp('MCMC optimisation process');
            disp('-------------------------');

            % loop (multiple) proposal
            for ii = 1:fitting.repetition
            fprintf('Repetition #%i/%i \n',ii,fitting.repetition)

            % reset start point, over-dispersed in u space for ii > 1
            uCurr = uStart;
            if ii > 1 && fitting.overdisp > 0
                uCurr = uCurr + fitting.overdisp .* uWidth .* randn(size(uCurr),'like',uCurr);
                uCurr = max(uCurr,uLo); uCurr = min(uCurr,uHi);
            end
            if hasJac || ii > 1
                xCurr = this.transform_inverse(uCurr, method, lb, ub);
                xCurr = max(xCurr,lb); xCurr = min(xCurr,ub);
            else
                xCurr = xStart;
            end
            logLCurr = loglik(xCurr);
            if hasJac; logJCurr = this.transform_logjac(uCurr, method, lb, ub); end

            % initial proposal scale in u space: xStepSize / |dx/du| at the start point
            sigma = repmat(xStepsize, 1, Nv);
            if hasJac
                logJ0           = this.transform_logjac(uCurr, method, lb, ub);
                sigma(~isLinear,:) = min( sigma(~isLinear,:) ./ exp(logJ0(~isLinear,:)), uWidth(~isLinear) );
            end

            accWin  = zeros(Nblock, Nv, 'like', uCurr);     % acceptance counts in the current adaptation window
            accPost = zeros(Nblock, Nv, 'like', uCurr);     % acceptance counts after burn-in
            jAdapt  = 0;

            % adaptive covariance: running moments (double), acceptances since the accumulation start,
            % and the proposal factor lambda.*L (used once acPhase is true)
            if isACov
                acPhase = false;
                acN     = 0;
                acMean  = zeros(Nvar, Nv, 'double', 'gpuArray');
                acM2    = zeros(Nvar, Nvar, Nv, 'double', 'gpuArray');
                acAccN  = zeros(1, Nv, 'like', uCurr);
                acValid = false(1, Nv, 'gpuArray');
                acLam   = [];  acL = [];  acLprop = [];
            end

            counter = 0; start = tic;
            for k = 1:fitting.iteration
                if ~isComponent
                    % ========== joint update: all parameters of a voxel together ==========
                    % 1. make a proposal with normal distribution in u space
                    if isACov && acPhase
                        uProposed   = uCurr + mcmc.acov_step(acLprop, randn(size(uCurr),'like',uCurr));
                    else
                        uProposed   = uCurr + sigma.*randn(size(uCurr),'like',uCurr);
                    end
                    % back to native space; find proposal that is out of bound for exclusion
                    if hasJac
                        % fused inverse transform and log-Jacobian (one GPU kernel)
                        [xProposed, logJProposed] = this.transform_inverse_logjac_fused(uProposed, code, lb, ub);
                    else
                        xProposed = uProposed;
                    end
                    if allReject
                        isOutofbound = max(or(xProposed<lb, xProposed>ub),[],1);
                    else
                        isOutofbound = max(isRejectRow & or(xProposed<lb, xProposed>ub),[],1);
                    end
                    % replace out of bound by boundary values to avoid error when computing probability
                    xProposed = max(xProposed,lb); xProposed = min(xProposed,ub);

                    % 2. Metropolis sampling
                    % 2.1 proposal probability (+ log-Jacobian of the transform)
                    logLProposed = loglik(xProposed);
                    % 2.2 accept with probability min(1, exp(logRatio)); NaN is rejected
                    if hasJac
                        logRatio            = logLProposed - logLCurr + sum(logJProposed - logJCurr, 1);
                        isAccepted          = exp(logRatio) > rand(1,Nv,'like',logLProposed);
                    else
                        % all linear: same expression as metropolis_hastings (the GPU evaluates fused
                        % elementwise expressions slightly differently, so keep it verbatim)
                        acceptanceRatio     = min(exp(logLProposed-logLCurr), 1);
                        isAccepted          = acceptanceRatio > rand(1,Nv,'like',logLProposed);
                        isOutofbound        = isOutofbound | isnan(logLProposed);  % min(NaN,1) = 1 would accept NaN
                    end
                    isAccepted(isOutofbound)= 0;    % reject out of bound (and NaN) proposal
                    % 2.3 update parameters if accepted
                    logLCurr(isAccepted)    = logLProposed(isAccepted);
                    uCurr(:,isAccepted)     = uProposed(:,isAccepted);
                    % all linear: x == u, only u is tracked in the joint loop
                    if hasJac
                        xCurr(:,isAccepted)     = xProposed(:,isAccepted);
                        logJCurr(:,isAccepted)  = logJProposed(:,isAccepted);
                    end

                else
                    % ========== componentwise update: one parameter at a time ==========
                    isAccepted = false(Nvar, Nv, 'like', isRejectRow);
                    for kp = 1:Nvar
                        % 1. proposal for parameter kp only
                        uProposed_p     = uCurr(kp,:) + sigma(kp,:).*randn(1,Nv,'like',uCurr);
                        if isLinear(kp)
                            xProposed_p = uProposed_p;
                        else
                            [xProposed_p, logJProposed_p] = this.transform_inverse_logjac_fused(uProposed_p, code(kp), lb(kp), ub(kp));
                        end
                        isOutofbound    = isRejectRow(kp) & or(xProposed_p<lb(kp), xProposed_p>ub(kp));
                        xProposed_p     = max(xProposed_p,lb(kp)); xProposed_p = min(xProposed_p,ub(kp));
                        xProposed       = xCurr; xProposed(kp,:) = xProposed_p;

                        % 2. Metropolis sampling, the cached loglik is the current state
                        logLProposed    = loglik(xProposed);
                        logRatio        = logLProposed - logLCurr;
                        if ~isLinear(kp); logRatio = logRatio + logJProposed_p - logJCurr(kp,:); end
                        isAccepted_p                = exp(logRatio) > rand(1,Nv,'like',logLProposed);
                        isAccepted_p(isOutofbound)  = 0;
                        % 3. update parameter kp and the cache before the next parameter
                        logLCurr(isAccepted_p)      = logLProposed(isAccepted_p);
                        uCurr(kp,isAccepted_p)      = uProposed_p(isAccepted_p);
                        xCurr(kp,isAccepted_p)      = xProposed_p(isAccepted_p);
                        if ~isLinear(kp); logJCurr(kp,isAccepted_p) = logJProposed_p(isAccepted_p); end
                        isAccepted(kp,:)            = isAccepted_p;
                    end
                end

                % 4. acceptance bookkeeping and adaptation (burn-in only, frozen afterwards)
                if k <= Nburnin
                    if isAdapt
                        accWin = accWin + isAccepted;
                        % adaptive covariance: accumulate the state after this iteration (after the warm-up)
                        if isACov && k > acWarm
                            [acMean, acM2, acN] = this.welford_update(acMean, acM2, acN, uCurr);
                            acAccN  = acAccN + isAccepted;
                        end
                        if mod(k, fitting.adaptInterval) == 0
                            jAdapt  = jAdapt + 1;
                            delta   = this.adapt_gain(jAdapt) .* (accWin./fitting.adaptInterval - fitting.adaptTarget);
                            if isACov && acPhase
                                % covariance phase: Robbins-Monro on log lambda (one scalar per voxel)
                                acLam   = acLam .* exp(delta);
                            else
                                % joint: delta is [1,Nv] and scales all parameters of a voxel together
                                sigma   = sigma .* exp(delta);
                                if hasJac; sigma(~isLinear,:) = min(sigma(~isLinear,:), uWidth(~isLinear)); end
                            end
                            accWin(:) = 0;
                            % adaptive covariance: switch (first time) and refresh C_i, L_i
                            if isACov && k >= kSwitch
                                if ~acPhase
                                    % previous factor at the switch: the diagonal step, lambda = 2.38/sqrt(d)
                                    acL     = eye(Nvar, 'like', uCurr) .* reshape(sigma ./ acScale, 1, Nvar, Nv);
                                    acLam   = acScale .* ones(1, Nv, 'like', uCurr);
                                    acPhase = true;
                                end
                                [acL, acValid] = this.acov_refresh(acM2, acN, acAccN, acL, acValid, acEps, acTiny);
                            end
                            if isACov && acPhase
                                acLprop = reshape(acLam, 1, 1, Nv) .* acL;
                            end
                        end
                    end
                else
                    accPost = accPost + isAccepted;
                end

                % 5. Maintain the independence between iterations
                % 5.1 discard the first burnin*100% iterations
                % 5.2 keep an iteration every N iterations
                if ( k > Nburnin ) && mod(k-Nburnin+1, fitting.thinning) == 0
                    counter = counter+1;
                    if hasJac; xPosterior(:,:,counter,ii) = gather(xCurr); else; xPosterior(:,:,counter,ii) = gather(uCurr); end
                end

                % display message at 1000 iteration and every 10000 iteration
                if mod(k,fitting.iteration/50) == 0 || k == min(1e3, fitting.iteration/100)
                    ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
                    ERT = ET / (k/fitting.iteration) - ET;
                    fprintf('Iteration #%6d,    Elapsed time (hh:mm:ss):%s,     Estimated remaining time (hh:mm:ss):%s \n',k,string(ET),string(ERT));
                end
            end

            acceptance(:,:,ii)  = gather(accPost.' ./ (fitting.iteration - Nburnin));
            stepSize(:,:,ii)    = gather(sigma.');
            if isACov
                % final (frozen) proposal covariance lambda^2 L L' (the diagonal one if never switched)
                if acPhase
                    Pc = gather(pagemtimes(acLprop, 'none', acLprop, 'transpose'));
                else
                    Pc = gather(eye(Nvar, 'like', uCurr) .* reshape(sigma.^2, 1, Nvar, Nv));
                    acLam = ones(1, Nv, 'like', uCurr);
                end
                acPropCov(:,:,:,ii) = permute(Pc, [3 1 2]);
                dPc                 = reshape(Pc, Nvar*Nvar, Nv);
                stepSize(:,:,ii)    = sqrt(dPc(1:Nvar+1:end, :)).';
                sv                  = pagesvd(double(Pc));                  % [d,1,Nv], descending
                acCond(:,ii)        = reshape(sv(1,1,:) ./ sv(end,1,:), Nv, 1);
                acLambda(:,ii)      = gather(acLam(:));
                acValidOut(:,ii)    = gather(acValid(:));
            end
            end

            % convert final posterior distribution into structure
            xPosterior = this.array2struct(xPosterior,fitting.modelParams);
            for kvar = 1:Nvar; xPosterior.(fitting.modelParams{kvar}) = shiftdim(xPosterior.(fitting.modelParams{kvar}),1); end

            % diagnostics and resolved settings
            diagnostics.acceptance  = acceptance;
            if isComponent; diagnostics.acceptanceBlocks = fitting.modelParams(:).'; else; diagnostics.acceptanceBlocks = {'joint'}; end
            diagnostics.stepSize        = stepSize;
            diagnostics.sampledParams   = fitting.modelParams(:).';
            if isACov
                diagnostics.adaptCovariance = struct('params', {fitting.modelParams(:).'}, ...
                    'proposalCov', acPropCov, 'lambda', acLambda, 'condition', acCond, 'valid', acValidOut, ...
                    'switchIteration', kSwitch);
                acRule = mcmc.acov_rule(acWarm, acMinN, kSwitch, acScale);
            else
                acRule = [];
            end
            diagnostics.settings    = struct( ...
                'parameterTransform',   {method}, ...
                'updateScheme',         lower(fitting.updateScheme), ...
                'adaptStepSize',        isAdapt, ...
                'adaptInterval',        fitting.adaptInterval, ...
                'adaptTarget',          fitting.adaptTarget, ...
                'adaptRule',            'log(sigma) += 2*j^(-0.6)*(acc_j - adaptTarget), every adaptInterval iterations within burn-in', ...
                'Nadapt',               Nadapt*isAdapt, ...
                'adaptCovariance',      isACov, ...
                'adaptCovarianceRule',  acRule, ...
                'Nburnin',              Nburnin, ...
                'overdisp',             fitting.overdisp, ...
                'overdispRule',         'u0 + overdisp*W.*randn for repetition > 1, W = T(ub)-T(lb)', ...
                'stepSizeInit',         'xStepSize ./ |dx/du| at the start point (u space)', ...
                'rngState',             rngState, ...
                'gpuRngState',          gpuRngState);

            disp('The Metropolis-Hastings MCMC sampling (adaptive) is completed.')

        end

        function xPosterior = goodman_weare(this,y,x0,weights,fitting,modelFWD,varargin)
        % Input
        % ----------
        % y         : measurements, [Nmeas,Nvoxels]
        % x0        : structure array, starting points, N fields, each field 1xNvoxel
        % weights   : weighting for non-linear least square fitting, same dimension as y
        % pars0         : Structure variable containing all parameters to be estimated
        % fitting       : Structure variable containing all fitting algorithm setting
        %   .modelParams       : 1xM cell variable,    name of the model parameters, e.g. {'S0','R2star','noise'};
        %   .lb                 : 1xM numeric variable, fitting lower bound, same order as field 'modelParams', e.g. [0.5, 0, 0.001];
        %   .ub                 : 1xM numeric variable, fitting upper bound, same order as field 'modelParams', e.g. [2, 1, 0.1];
        %   .iteration          : # MCMC iterations
        %   .thinning           : sampling interval between iterations
        %   .burnin             : iterations at the beginning to be discarded, if burnin>1, then the exact number will  be used; if 0<burnin<1 then actual burnin = iteration*burnin
        %   .repetition         : # repetition of MCMC proposal
        %   .StepSize           : step size for 'GW' in MCMC proposal ('GW' only)
        %   .Nwalker            : # random walkers ('GW' only)
        %   .Ensembleupdate    : (optional) ensemble update scheme:
        %                          'simultaneous' - original behaviour (DEFAULT, backward compatible):
        %                                           all walkers proposed/updated in one pass using a
        %                                           single derangement as partners.
        %                          'redblack'     - affine-invariance-correct parallel update
        %                                           (Foreman-Mackey et al. 2013): split the ensemble
        %                                           into two halves and update each half using the
        %                                           other (frozen) half as anchors.
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
            Nvar     = numel(fitting.modelParams);
            Nburnin  = this.get_number_burnin(fitting);
            % Ns: # samples in posterior distribution
            Ns       = numel(Nburnin+1:fitting.thinning:fitting.iteration);
            Nwalker  = fitting.Nwalker;
            StepSize = fitting.StepSize;

            % red-black update scheme
            useRedBlack = strcmpi(fitting.Ensembleupdate,'redblack');
            if useRedBlack
                if Nwalker < 2
                    error('goodman_weare:Nwalker', ...
                        'redblack update needs Nwalker >= 2 (use even, >= 2*Nvar in practice).');
                end
                if mod(Nwalker,2) ~= 0
                    warning('goodman_weare:Nwalker', ...
                        'Nwalker is odd; redblack halves will be unequal. Even Nwalker is conventional.');
                end
                h       = floor(Nwalker/2);
                halfIdx = {1:h, h+1:Nwalker};   % {S0, S1}: complementary anchor sets
            end
            fprintf('Ensemble update scheme: %s\n', fitting.Ensembleupdate);
        
            % convert data into single datatype for better performance
            y       = gpuArray( single(y) );
            weights = gpuArray( single(weights) );
            for km = 1:Nvar; x0.(fitting.modelParams{km}) = gpuArray(single( x0.(fitting.modelParams{km}) ));end
        
            % setup boundary variables
            lb          = gpuArray( single(repmat(fitting.lb(:),1,Nv,Nwalker)));
            ub          = gpuArray( single(repmat(fitting.ub(:),1,Nv,Nwalker)));
            % set up weight
            weights     = repmat(weights,1,1,Nwalker);
            % initialize array to staore all the samples
            xPosterior  = zeros(Nvar, Nv, Nwalker, Ns, fitting.repetition,'single');
            
            % initiate an ensemble of walkers around the starting position (0.1% full range) with Gaussian distribution
            % 1st: Nvar;2nd: Nv; 3rd: Nwalker
            xCurr   = this.struct2array(x0,fitting.modelParams);                % extract parameter structure to numeric array for faster computation
            xCurr   = xCurr + (ub-lb)*fitting.startRange.*randn(size(ub));     % initiate starting position for all walkers
            xCurr   = max(xCurr,lb); xCurr = min(xCurr,ub);                     % set boundary
            x0      = this.array2struct(xCurr,fitting.modelParams);             % convert array back to structure for FWD function
            % compute likelihood at starting points
            logP0   = arrayfun(@logP_Gaussian, sum( weights.* (modelFWD(x0, varargin{:})-y).^2, 1 ), x0.noise, Nm);
        
            disp('-------------------------');
            disp('MCMC optimisation process');
            disp('-------------------------');
        
            for ii = 1:fitting.repetition
            fprintf('Repetition #%i/%i \n',ii,fitting.repetition)
        
            logPCurr= logP0;
            xCurr   = this.struct2array(x0,fitting.modelParams);
        
            counter = 0; start = tic;
            for k = 1:fitting.iteration

                if ~useRedBlack
                    % =========== LEGACY: simultaneous single-derangement update ===========
                    % (identical to the original implementation; preserved for reproducibility)
                    % 1.1 find a unique partner for each walker

                    % 1. make a proposal with normal distribution
                    % 1.1. find a unique partner for a walker k
                    partner         = this.find_partner(Nwalker);
                    % 1.2. stretch move
                    zz              = ((StepSize-1)*rand(size(logP0),'like',xCurr) + 1).^2 / StepSize;
                    xProposed       = xCurr(:,:,partner) + (xCurr - xCurr(:,:,partner)).*zz;
                    % find proposal that is out of bound for exclusion
                    isOOB    = max(or(xProposed<lb, xProposed>ub),[],1);    
                    % replace boundary values so it does not give error when compting probability
                    xProposed = max(xProposed,lb); xProposed = min(xProposed,ub);
                    % convert the proposal into structure array for FWD function
                    xProposed_struct = this.array2struct(xProposed,fitting.modelParams);
            
                    % 2. Metropolis sampling
                    % If the probability ratio of new to old > threshold, we take the new solution.
                    % 2.1 proposal probability
                    logPProposed            = arrayfun(@logP_Gaussian, sum( weights.* (modelFWD(xProposed_struct, varargin{:})-y).^2, 1 ), xProposed_struct.noise, Nm);
                    % 2.2 Compute acceptance ratio based on z^(Nd-1)*new/old probability
                    acceptanceRatio         = min(zz.^(Nvar-1).*exp(logPProposed-logPCurr), 1);
                    isAccepted              = acceptanceRatio > rand(1,Nv,Nwalker,'like',xCurr);
                    isAccepted(isOOB)       = 0;    % reject out of bound proposal
                    % 2.3 update parameters
                    logPCurr(isAccepted)    = logPProposed(isAccepted);
                    xCurr(:,isAccepted)     = xProposed(:,isAccepted);
                else

                    % =========== RED-BLACK: two complementary sub-steps per sweep ===========
                    for s = 1:2
                        active = halfIdx{s};       % walkers moved this sub-step
                        frozen = halfIdx{3-s};     % anchors, held FIXED this sub-step
                        na     = numel(active);

                        % stretch proposal: each active walker anchored on a RANDOM frozen
                        % walker (uniform, WITH replacement -- textbook GW partner draw)
                        pidx  = frozen(randi(numel(frozen),1,na));
                        xAnch = xCurr(:,:,pidx);                 % [Nvar,Nv,na] fixed
                        xAct  = xCurr(:,:,active);               % [Nvar,Nv,na]
                        zz    = ((StepSize-1)*rand(1,Nv,na,'like',xCurr) + 1).^2 / StepSize;
                        xProp = xAnch + (xAct - xAnch).*zz;

                        lb_a  = lb(:,:,active);  ub_a = ub(:,:,active);
                        isOOB = max(or(xProp<lb_a, xProp>ub_a),[],1);
                        xProp = max(xProp,lb_a);  xProp = min(xProp,ub_a);

                        % proposal likelihood on the active half only
                        % (two half-width evals/iter ~= one full eval: no cost regression)
                        xProp_s  = this.array2struct(xProp,fitting.modelParams);
                        logPProp = arrayfun(@logP_Gaussian, ...
                                     sum( weights(:,:,active).*(modelFWD(xProp_s,varargin{:})-y).^2, 1 ), ...
                                     xProp_s.noise, Nm);

                        % acceptance  z^(Nvar-1) * pi(new)/pi(old)
                        logPAct  = logPCurr(:,:,active);
                        aR       = min(zz.^(Nvar-1).*exp(logPProp - logPAct), 1);
                        acc      = aR > rand(1,Nv,na,'like',xCurr);
                        acc(isOOB) = 0;

                        % scatter accepted moves back into the active block
                        xAct(:,acc)  = xProp(:,acc);
                        logPAct(acc) = logPProp(acc);
                        xCurr(:,:,active)    = xAct;
                        logPCurr(:,:,active) = logPAct;
                    end
                end
        
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
            xPosterior = this.array2struct(xPosterior,fitting.modelParams);
            for kvar = 1:Nvar; xPosterior.(fitting.modelParams{kvar}) = shiftdim(xPosterior.(fitting.modelParams{kvar}),1); end
        
            disp('The affine-invariant ensemble MCMC sampling is completed.')
        
        end

        function xPosterior = goodman_weare_wglobal_constant(this,y,x0,weights,fitting,modelFWD,varargin)
        % Affine-invariant ensemble sampling for hierarchical / partial-pooling
        % problems with GLOBAL (shared) parameters, via block Metropolis-within-Gibbs.
        %
        % CONTRACT (scope of this sampler):
        %   The log-likelihood must factorise over units (voxels) as
        %       sum_z loglik(y_z | theta_z, phi)
        %   i.e. given the global parameters phi, the units are conditionally
        %   independent. Parameters split into LOCAL theta_z (one set per unit) and
        %   GLOBAL phi (shared). Any model with this structure is supported
        %   (arbitrary NND local and NGlobal global parameters); the sampler holds
        %   no model-specific knowledge.
        %
        % WHICH PARAMETERS ARE GLOBAL is detected from the starting structure: a
        % field of x0 with size==1 along the voxel dimension is treated as global
        % (see check_global_constant). So {R2c,k2A}, a shared diffusivity, a global
        % B1 term, etc., all work without code changes.
        %
        % One iteration = two decoupled blocks, each a Goodman-Weare stretch move:
        %   LOCAL block : propose theta, GLOBALS HELD FIXED, exponent z^(NND-1),
        %                 forward model evaluated at (new local, old global).
        %   GLOBAL block: propose phi, LOCALS HELD FIXED at their updated values,
        %                 exponent z^(NGlobal-1), forward model evaluated at
        %                 (current local, new global); acceptance uses the summed
        %                 log-likelihood DIFFERENCE accumulated in double precision.
        %
        % This structure is what keeps the cached likelihood consistent with the
        % current state at all times (the previous single-eval / two-accept scheme
        % left it stale) and gives each block its correct stretch Jacobian.
        %
        % NOTES / options:
        %   .StepSize          : stretch scale 'a'
        %   .Nwalker           : # walkers (even, >= 2*max(NND,NGlobal) recommended)
        %   .startRange        : local walker init spread (fraction of range)
        %   .globalWarmup      : (optional) # global-only iterations before the joint
        %                        loop (locals frozen). Default 0. Helps when the global
        %                        posterior is very tight.
        %   .Ensembleupdate          : ensemble update scheme, applied to BOTH blocks:
        %                          'simultaneous' - (DEFAULT, backward compatible)
        %                                           all walkers proposed at once using a
        %                                           single derangement as partners.
        %                          'redblack'     - correct parallel update
        %                                           (Foreman-Mackey 2013): split walkers
        %                                           into two halves and update each half
        %                                           using the other as frozen anchors.
        %                        Consistent with the option in goodman_weare.
        %
        % Goodman & Weare 2010; emcee (Foreman-Mackey 2013).

            fitting = this.check_set_default_basic(fitting);
            if isempty(weights); weights = ones(size(y), 'like', y); end

            [Nm, Nv] = size(y);
            Nvar     = numel(fitting.modelParams);
            Nburnin  = this.get_number_burnin(fitting);
            Ns       = numel(Nburnin+1:fitting.thinning:fitting.iteration);
            Nwalker  = fitting.Nwalker;
            StepSize = fitting.StepSize;

            % --- ensemble update scheme (consistent with goodman_weare) -----
            useRedBlack = strcmpi(fitting.Ensembleupdate,'redblack');
            if useRedBlack
                if mod(Nwalker,2) ~= 0
                    warning('goodman_weare_wglobal_constant:Nwalker', ...
                        'Nwalker is odd; red-black halves will be unequal. Even Nwalker is conventional.');
                end
                h       = floor(Nwalker/2);
                halfIdx = {1:h, h+1:Nwalker};
            end
            fprintf('Ensemble update scheme: %s\n', fitting.Ensembleupdate);
            % -----------------------------------------------------------------

            % convert to single + push to GPU
            y       = gpuArray( single(y) );
            weights = gpuArray( single(weights) );
            for km = 1:Nvar; x0.(fitting.modelParams{km}) = gpuArray(single( x0.(fitting.modelParams{km}) )); end

            isGlobal = this.check_global_constant(x0,fitting.modelParams);
            NGlobal  = numel(isGlobal(isGlobal==1));
            NND      = Nvar - NGlobal;
            if NND==0 || NGlobal==0
                error('goodman_weare_wglobal_constant:partition', ...
                    'Need at least one local and one global parameter (NND=%d, NGlobal=%d). Use goodman_weare for the all-local case.',NND,NGlobal);
            end

            % bounds
            lb        = gpuArray( single(repmat(fitting.lb(isGlobal==0),1,Nv,Nwalker)));
            ub        = gpuArray( single(repmat(fitting.ub(isGlobal==0),1,Nv,Nwalker)));
            lb_global = gpuArray( single(repmat(fitting.lb(isGlobal==1),1,1,Nwalker)));
            ub_global = gpuArray( single(repmat(fitting.ub(isGlobal==1),1,1,Nwalker)));
            weights   = repmat(weights,1,1,Nwalker);

            xPosterior_ND     = zeros(NND, Nv, Nwalker, Ns, fitting.repetition,'single');
            xPosterior_global = zeros(NGlobal, 1, Nwalker, Ns, fitting.repetition,'single');

            % --- initialise walkers -------------------------------------------------
            [xCurr_ND,xCurr_global] = this.struct2array_wglobal(x0,fitting.modelParams);
            % local: tight cloud around starting point
            xCurr_ND     = xCurr_ND + (ub-lb)*fitting.startRange.*randn(size(ub));
            xCurr_ND     = min(max(xCurr_ND,lb),ub);
            % GLOBAL: OVER-DISPERSED across the full prior box. Ensemble samplers
            % cannot manufacture spread a collapsed cloud lacks, and the global
            % posterior is typically very tight (informed by all voxels), so a small
            % cloud can get stuck. Over-dispersion contracts correctly; collapse does not.
            % xCurr_global = lb_global + (ub_global-lb_global).*rand(size(ub_global));
            xCurr_global = xCurr_global + (ub_global-lb_global)*fitting.startRange.*rand(size(ub_global));
            x0 = this.array2struct_wglobal(xCurr_ND,xCurr_global,fitting.modelParams,isGlobal);

            logP0 = arrayfun(@logP_Gaussian, sum( weights.* (modelFWD(x0,varargin{:})-y).^2, 1 ), x0.noise, Nm);

            disp('-------------------------');
            disp('MCMC optimisation process (block GW: local + global)');
            disp('-------------------------');

            for ii = 1:fitting.repetition
                fprintf('Repetition #%i/%i \n',ii,fitting.repetition)
    
                logPCurr = logP0;
                [xCurr_ND,xCurr_global] = this.struct2array_wglobal(x0,fitting.modelParams);

            % optional global-only warmup (locals frozen) to settle the tight global block
            for kw = 1:fitting.globalWarmup
                [xCurr_global,logPCurr] = this.gw_global_block( ...
                    xCurr_ND,xCurr_global,logPCurr,weights,y,Nm,Nv,Nwalker, ...
                    StepSize,NGlobal,lb_global,ub_global,fitting,isGlobal,modelFWD,varargin{:});
            end

            counter = 0; start = tic;
            for k = 1:fitting.iteration

                % ============ LOCAL block (globals fixed) ============
                if ~useRedBlack
                    % --- simultaneous (default, original behaviour) ---
                    partner  = this.find_partner(Nwalker);
                    zz_ND    = ((StepSize-1)*rand(1,Nv,Nwalker,'like',xCurr_ND) + 1).^2 / StepSize;
                    xProp_ND = xCurr_ND(:,:,partner) + (xCurr_ND - xCurr_ND(:,:,partner)).*zz_ND;
                    oob_ND   = max(or(xProp_ND<lb, xProp_ND>ub),[],1);
                    xProp_ND = min(max(xProp_ND,lb),ub);
                    s_local  = this.array2struct_wglobal(xProp_ND, xCurr_global, fitting.modelParams, isGlobal);
                    logP_loc = arrayfun(@logP_Gaussian, sum( weights.*(modelFWD(s_local,varargin{:})-y).^2, 1 ), s_local.noise, Nm);
                    aR       = min( zz_ND.^(NND-1).*exp(logP_loc - logPCurr), 1);
                    acc      = aR > rand(1,Nv,Nwalker,'like',xCurr_ND);
                    acc(oob_ND) = 0;
                    logPCurr(acc)   = logP_loc(acc);       % cache <- (new local, OLD global)
                    xCurr_ND(:,acc) = xProp_ND(:,acc);
                else
                    % --- red-black: two sub-steps over walker halves ---
                    for s = 1:2
                        active = halfIdx{s}; frozen = halfIdx{3-s}; na = numel(active);
                        pidx   = frozen(randi(numel(frozen),1,na));
                        xAnch  = xCurr_ND(:,:,pidx); xAct = xCurr_ND(:,:,active);
                        zz_ND  = ((StepSize-1)*rand(1,Nv,na,'like',xCurr_ND) + 1).^2 / StepSize;
                        xProp  = xAnch + (xAct - xAnch).*zz_ND;
                        lb_a   = lb(:,:,active); ub_a = ub(:,:,active);
                        isOOB  = max(or(xProp<lb_a, xProp>ub_a),[],1);
                        xProp  = min(max(xProp,lb_a),ub_a);
                        % eval at (new local active, OLD global active) -- globals fixed this block
                        s_loc  = this.array2struct_wglobal(xProp, xCurr_global(:,:,active), fitting.modelParams, isGlobal);
                        logP_loc = arrayfun(@logP_Gaussian, sum( weights(:,:,active).*(modelFWD(s_loc,varargin{:})-y).^2, 1 ), s_loc.noise, Nm);
                        logPAct  = logPCurr(:,:,active);
                        aR       = min( zz_ND.^(NND-1).*exp(logP_loc - logPAct), 1);
                        acc      = aR > rand(1,Nv,na,'like',xCurr_ND); acc(isOOB) = 0;
                        xAct(:,acc)  = xProp(:,acc);
                        logPAct(acc) = logP_loc(acc);
                        xCurr_ND(:,:,active)    = xAct;
                        logPCurr(:,:,active)    = logPAct;
                    end
                end

                % ============ GLOBAL block (locals fixed at updated values) ============
                [xCurr_global,logPCurr] = this.gw_global_block( ...
                    xCurr_ND,xCurr_global,logPCurr,weights,y,Nm,Nv,Nwalker, ...
                    StepSize,NGlobal,lb_global,ub_global,fitting,isGlobal,modelFWD,varargin{:});

                % thinning / burn-in
                if ( k > Nburnin ) && mod(k-Nburnin+1, fitting.thinning) == 0
                    counter = counter+1;
                    xPosterior_ND(:,:,:,counter,ii)     = gather(xCurr_ND);
                    xPosterior_global(:,:,:,counter,ii) = gather(xCurr_global);
                end

                if mod(k,fitting.iteration/50) == 0 || k == min(1e2, fitting.iteration/100)
                    ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
                    ERT = ET / (k/fitting.iteration) - ET;
                    fprintf('Iteration #%6d,    Elapsed time (hh:mm:ss):%s,     Estimated remaining time (hh:mm:ss):%s \n',k,string(ET),string(ERT));
                end
            end
            end

            xPosterior = this.array2struct_wglobal(xPosterior_ND,xPosterior_global,fitting.modelParams,isGlobal);
            for kvar = 1:Nvar; xPosterior.(fitting.modelParams{kvar}) = shiftdim(xPosterior.(fitting.modelParams{kvar}),1); end

            disp('The block affine-invariant ensemble MCMC sampling is completed.')

        end

        % ---- GLOBAL block: one GW stretch update of phi, locals held fixed -------
        function [xCurr_global,logPCurr] = gw_global_block(this, ...
                xCurr_ND,xCurr_global,logPCurr,weights,y,Nm,Nv,Nwalker, ...
                StepSize,NGlobal,lb_global,ub_global,fitting,isGlobal,modelFWD,varargin)

            useRedBlack = strcmpi(fitting.Ensembleupdate,'redblack');
            if ~useRedBlack
                % --- simultaneous (default, original behaviour) ---
                partner = this.find_partner(Nwalker);
                zz_g    = ((StepSize-1)*rand(1,1,Nwalker,'like',xCurr_global) + 1).^2 / StepSize;
                gProp   = xCurr_global(:,:,partner) + (xCurr_global - xCurr_global(:,:,partner)).*zz_g;
                oob_g   = max(or(gProp<lb_global, gProp>ub_global),[],1);
                gProp   = min(max(gProp,lb_global),ub_global);
                s_glob  = this.array2struct_wglobal(xCurr_ND, gProp, fitting.modelParams, isGlobal);
                logP_g  = arrayfun(@logP_Gaussian, sum( weights.*(modelFWD(s_glob,varargin{:})-y).^2, 1 ), s_glob.noise, Nm);
                dlogP   = sum( double(logP_g - logPCurr), 2 );
                aR_g    = min( zz_g.^(NGlobal-1).*exp(dlogP), 1 );
                acc_g   = aR_g > rand(1,1,Nwalker,'like',xCurr_global);
                acc_g(oob_g) = 0; acc_g = logical(acc_g);
                xCurr_global(:,:,acc_g) = gProp(:,:,acc_g);
                logPCurr(:,:,acc_g)     = logP_g(:,:,acc_g);
            else
                % --- red-black: two sub-steps over walker halves ---
                h_g = floor(Nwalker/2); halfIdx_g = {1:h_g, h_g+1:Nwalker};
                for s = 1:2
                    active = halfIdx_g{s}; frozen = halfIdx_g{3-s}; na = numel(active);
                    pidx   = frozen(randi(numel(frozen),1,na));
                    xg_act = xCurr_global(:,:,active);
                    zz_g   = ((StepSize-1)*rand(1,1,na,'like',xCurr_global) + 1).^2 / StepSize;
                    gProp  = xCurr_global(:,:,pidx) + (xg_act - xCurr_global(:,:,pidx)).*zz_g;
                    oob_g  = max(or(gProp<lb_global(:,:,active), gProp>ub_global(:,:,active)),[],1);
                    gProp  = min(max(gProp,lb_global(:,:,active)),ub_global(:,:,active));
                    % eval at (current local active, new global active) -- locals fixed this block
                    s_glob = this.array2struct_wglobal(xCurr_ND(:,:,active), gProp, fitting.modelParams, isGlobal);
                    logP_g = arrayfun(@logP_Gaussian, sum( weights(:,:,active).*(modelFWD(s_glob,varargin{:})-y).^2, 1 ), s_glob.noise, Nm);
                    lp_act = logPCurr(:,:,active);
                    dlogP  = sum( double(logP_g - lp_act), 2 );    % [1,1,na], double
                    aR_g   = min( zz_g.^(NGlobal-1).*exp(dlogP), 1 );
                    acc_g  = aR_g > rand(1,1,na,'like',xCurr_global);
                    acc_g(oob_g) = 0; acc_g = logical(acc_g);
                    xg_act(:,:,acc_g)  = gProp(:,:,acc_g);
                    lp_act(:,:,acc_g)  = logP_g(:,:,acc_g);
                    xCurr_global(:,:,active) = xg_act;
                    logPCurr(:,:,active)     = lp_act;
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
        %   .thinning   : MCMC thinning interval,           default = every 20 iterations
        %   .burnin     : MCMC burn-in ratio,               default = 10%
        %   .metric     : method to compute expected valur from posterior distribution, 'mean' (default) | 'median'
        %   .algorithm  : MCMC algorithm 'MH': Metropolis-Hastings; 'GW': Goodman-Weare, 'MH' (default) | 'GW' 
        %   .StepSize   : Step size for Goodman-Weare,      default = 2
        %   .Nwalker    : number of walkers for Goodman-Weare,      default = 50
        %
            fitting2 = fitting;

            % get fitting algorithm setting
            if ~isfield(fitting,'iteration');           fitting2.iteration      = 2e5;              end
            if ~isfield(fitting,'thinning');            fitting2.thinning       = 20;               end  % thinning, sampled every 100 interval
            if ~isfield(fitting,'metric');              fitting2.metric         = {'mean','std'};   end
            if ~isfield(fitting,'burnin');              fitting2.burnin         = 0.1;              end  % 10% burnin
            
            if ~isfield(fitting,'repetition');          fitting2.repetition     = 1;                end 
            if ~isfield(fitting,'outputFilename');      fitting2.outputFilename = [];               end
            if ~isfield(fitting,'algorithm');           fitting2.algorithm      = 'MH';             end
            if ~isfield(fitting,'StepSize');            fitting2.StepSize       = 2;                end
            if ~isfield(fitting,'Nwalker');             fitting2.Nwalker        = 50;               end 
            if ~isfield(fitting,'ub');                  fitting2.ub             = [];               end
            if ~isfield(fitting,'lb');                  fitting2.lb             = [];               end
            if ~isfield(fitting,'startRange');          fitting2.startRange     = 0.001;            end

            % --- choose ensemble update scheme (default = original behaviour) ----
            if ~isfield(fitting,'Ensembleupdate');      fitting2.Ensembleupdate  = 'simultaneous';  end
            if ~isfield(fitting,'globalWarmup');        fitting2.globalWarmup   = 0;          end

            if any(ismember(fitting2.metric,'mode'))
                if ~isfield(fitting,'Nbin');            fitting2.Nbin     = 1001;                end 
            end

            if ~isfield(fitting,'autoMemManage');          fitting2.autoMemManage     = true;            end
            

            if ~iscell(fitting2.metric)
                fitting2.metric = cellstr(fitting2.metric);
            end

            if strcmpi(fitting2.algorithm ,'gw'); fitting2.algorithm = 'ensemble'; end % legacy
        end

        % display fitting algorithm parameters
        function display_basic_algorithm_parameters(fitting)

            if strcmpi( fitting.algorithm, 'ensemble'); algorithm = 'Affine-Invariant Ensemble';
            else;                                       algorithm = 'Metropolis-Hastings';          end

            disp('----------------------------------------------------');
            disp('Markov Chain Monte Carlo (MCMC) algorithm parameters');
            disp('----------------------------------------------------');
            disp(['Algorithm         : ', algorithm]);
            disp(['No. of iterations : ', num2str(fitting.iteration)]);
            disp(['No. of repetitions: ', num2str(fitting.repetition)])
            disp(['Thinning          : ', num2str(fitting.thinning)]);
            disp(['Burn-in (#iter.)  : '  num2str(mcmc.get_number_burnin(fitting))])
            disp(['Metric(s)         : ', cell2str(fitting.metric)]);
            if strcmpi( fitting.algorithm, 'ensemble'); disp(['Step size         : ', num2str(fitting.StepSize) ]); end
            if strcmpi( fitting.algorithm, 'ensemble'); disp(['No. of walkers    : ', num2str(fitting.Nwalker) ]); end

        end
        
        % save the mcmc output structure variable into disk space 
        function save_mcmc_output(outputFilename,out)
        % Input
        % ------------------
        % outputFilename   : output filename
        % out               : output structure of askadam
        %

            % save the estimation results if the output filename is provided
            if ~isempty(outputFilename)
                [output_dir,~,~] = fileparts(outputFilename);
                if ~exist(output_dir,'dir')
                    mkdir(output_dir);
                end
                save(outputFilename,'out');
                fprintf('Estimation output is saved at %s\n',outputFilename);
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

        function x_struct = array2struct_wglobal(x_ND,x_global,fields, isGlobal)

            ctr_global = 1; ctr_ND = 1;
            for k = 1:numel(fields)
                if isGlobal(k)
                     x_struct.(fields{k}) = x_global(ctr_global,:,:,:,:,:);
                     ctr_global = ctr_global + 1;
                else
                    x_struct.(fields{k}) = x_ND(ctr_ND,:,:,:,:,:);
                    ctr_ND = ctr_ND +1;
                end
            end
        end

        function [x_ND,x_global] = struct2array_wglobal(x_struct,fields)

            % check if the fitting parameter is global or voxel
            isGlobal = mcmc.check_global_constant(x_struct,fields);
            nVol = 0; for k = 1:numel(fields); nVol = max(size(x_struct.(fields{k}),2),nVol); end
       
            nWalker     = size(x_struct.(fields{1}),3);
            x_ND        = gpuArray(zeros(numel(isGlobal(isGlobal==0)),nVol,nWalker,"single"));
            x_global    = gpuArray(zeros(numel(isGlobal(isGlobal==1)),1,nWalker,"single"));

            ctr_global = 1; ctr_ND = 1;
            for k = 1:numel(fields)
                if isGlobal(k)
                    x_global(ctr_global,1,:)    = x_struct.(fields{k});
                    ctr_global                  = ctr_global +1;
                else
                    x_ND(ctr_ND,:,:)    = x_struct.(fields{k});
                    ctr_ND              = ctr_ND +1;
                end
            end
        end

        function isGlobal = check_global_constant(x_struct,fields)
            % check if the fitting parameter is global or voxel
            isGlobal = zeros(numel(fields),1); 
            for k = 1:numel(fields)
                if size(x_struct.(fields{k}),2) > 1
                    isGlobal(k) = false;
                else
                    isGlobal(k) = true;
                end
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

        % convert estimation into organised output structure
        function out = res2out(xPosterior,fitting,mask)
            
            % store the unshaped posterior into out
            out.posterior = xPosterior;

            % compute additional metric if specified
            fields = fieldnames(xPosterior);

            % Nvox    = size(xPosterior.(fields{1}),1);
            Nsample = prod(size(xPosterior.(fields{1}),2:5));

            metrics = fitting.metric;
            if ~isempty(metrics)
                for km = 1:numel(metrics)
                    switch lower(metrics{km})
                        case 'mean'
                            for kvar=1:numel(fields)
                                tmp = mean( reshape( xPosterior.(fields{kvar}), [size(xPosterior.(fields{kvar}),1), Nsample]),2);
                                tmp = utils.reshape_ND2image(tmp,mask);
                                out.mean.(fields{kvar}) = tmp;
                            end
                        case 'median'
                            for kvar=1:numel(fields)
                                tmp = median( reshape( xPosterior.(fields{kvar}), [size(xPosterior.(fields{kvar}),1), Nsample]),2);
                                tmp = utils.reshape_ND2image(tmp,mask);
                                out.median.(fields{kvar}) = tmp;
                            end
                        case 'std'
                            for kvar=1:numel(fields)
                                tmp = std( reshape( xPosterior.(fields{kvar}), [size(xPosterior.(fields{kvar}),1), Nsample]),[],2);
                                tmp = utils.reshape_ND2image(tmp,mask);
                                out.std.(fields{kvar}) = tmp;
                            end
                        case 'iqr'
                            for kvar=1:numel(fields)
                                tmp = iqr( reshape( xPosterior.(fields{kvar}), [size(xPosterior.(fields{kvar}),1), Nsample]),2);
                                tmp = utils.reshape_ND2image(tmp,mask);
                                out.iqr.(fields{kvar}) = tmp;
                            end
                        case 'mode'
                            for kvar=1:numel(fields)

                                Nbin = fitting.Nbin;

                                idx     = find(ismember(fitting.modelParams,fields{kvar}));
                                edges   = linspace(fitting.lb(idx)-1e-8,fitting.ub(idx)+1e-8,Nbin);

                                tmp     = reshape( xPosterior.(fields{kvar}), [size(xPosterior.(fields{kvar}),1), Nsample]);
                                tmp     = mode(discretize(tmp,edges),2);
                                tmp     = (edges(tmp) + edges(tmp+1)) / 2;
                                tmp     = utils.reshape_ND2image(tmp.',mask);
                                out.mode.(fields{kvar}) = tmp;
                            end
                    end
                end
            end
        end

        % make sure all network parameters stay between 0 and 1
        function parameters = set_boundary(parameters,ub,lb)

            field = fieldnames(parameters);
            for k = 1:numel(field)
                parameters.(field{k})   = max(parameters.(field{k}),lb(k)); % Lower bound     
                parameters.(field{k})   = min(parameters.(field{k}),ub(k)); % upper bound

            end

        end

        %% opt-in sampler options (metropolis_hastings_adaptive; also used by mcmc_bayes)
        % option name and legacy default of the opt-in sampler options
        function defaults = sampler_infra_defaults()
            defaults = { 'parameterTransform',  'linear';
                         'updateScheme',        'joint';
                         'adaptStepSize',       false;
                         'adaptInterval',       50;
                         'adaptTarget',         [];
                         'adaptCovariance',     false;
                         'overdisp',            0};
        end

        % names of the options in fitting that are set to a non-default value
        function nonDefault = nondefault_options(fitting, defaults)
        % Input
        % -----
        % fitting       : fitting structure
        % defaults      : {name, legacy default} rows; [] default: the option must be empty,
        %                 char default: case-insensitive match (a cell must match in every entry),
        %                 numeric/logical default: a scalar equal to it
        % Output
        % ------
        % nonDefault    : cell array of names of the non-default option(s)
        %
            nonDefault = {};
            if isempty(fitting) || ~isstruct(fitting); return; end

            for k = 1:size(defaults,1)
                name = defaults{k,1};
                if ~isfield(fitting,name); continue; end

                val = fitting.(name);
                ref = defaults{k,2};

                if isempty(ref)
                    isDefault = isempty(val);
                elseif ischar(ref)
                    if iscell(val)  % per-parameter setting, e.g. parameterTransform
                        isDefault = ~isempty(val) && all(cellfun(@(x) (ischar(x)||isstring(x)) && strcmpi(x,ref), val));
                    else
                        isDefault = (ischar(val)||isstring(val)) && strcmpi(val,ref);
                    end
                else
                    isDefault = (isnumeric(val)||islogical(val)) && isscalar(val) && val == ref;
                end

                if ~isDefault; nonDefault{end+1} = name; end %#ok<AGROW>
            end
        end

        % true if any opt-in sampler option is set to a non-default value
        function tf = use_sampler_infra(fitting)
            tf = ~isempty(mcmc.nondefault_options(fitting, mcmc.sampler_infra_defaults()));
        end

        % defaults and validation of the opt-in sampler options (see the class header)
        function fitting2 = check_set_default_infra(fitting)
            fitting2 = fitting;

            if ~isfield(fitting,'parameterTransform');  fitting2.parameterTransform = 'linear';      end
            if ~isfield(fitting,'updateScheme');        fitting2.updateScheme       = 'joint';       end
            if ~isfield(fitting,'adaptStepSize');       fitting2.adaptStepSize      = false;         end
            if ~isfield(fitting,'adaptInterval');       fitting2.adaptInterval      = 50;            end
            if ~isfield(fitting,'adaptTarget');         fitting2.adaptTarget        = [];            end
            if ~isfield(fitting,'adaptCovariance');     fitting2.adaptCovariance    = false;         end
            if ~isfield(fitting,'overdisp');            fitting2.overdisp           = 0;             end

            if ~any(strcmpi(fitting2.updateScheme,{'joint','componentwise'}))
                error('mcmc:invalidUpdateScheme', ...
                    'mcmc: fitting.updateScheme must be ''joint'' or ''componentwise'' (got ''%s'').', char(fitting2.updateScheme));
            end
            % target acceptance rate per update scheme
            if isempty(fitting2.adaptTarget)
                if strcmpi(fitting2.updateScheme,'componentwise'); fitting2.adaptTarget = 0.44; else; fitting2.adaptTarget = 0.234; end
            end
            if isempty(fitting2.overdisp); fitting2.overdisp = 0; end

            % adaptive covariance: joint update with step-size adaptation only
            ac = fitting2.adaptCovariance;
            if isempty(ac); ac = false; end
            if ~((islogical(ac) || isnumeric(ac)) && isscalar(ac) && any(double(ac) == [0 1]))
                error('mcmc:adaptCovariance', 'mcmc: fitting.adaptCovariance must be true or false.');
            end
            fitting2.adaptCovariance = logical(ac);
            if fitting2.adaptCovariance && ~strcmpi(fitting2.updateScheme, 'joint')
                error('mcmc:adaptCovariance', ...
                    ['mcmc: fitting.adaptCovariance = true needs updateScheme = ''joint'' (got ''%s''): the ' ...
                     'adaptive covariance proposal moves all sampled parameters of a voxel together.'], char(fitting2.updateScheme));
            end
            if fitting2.adaptCovariance && ~(fitting2.adaptStepSize)
                error('mcmc:adaptCovariance', ...
                    ['mcmc: fitting.adaptCovariance = true needs adaptStepSize = true (the covariance and its ' ...
                     'scale are learnt during burn-in at the adaptation steps).']);
            end
        end

        % display the opt-in sampler settings
        function display_infra_algorithm_parameters(fitting)
            method = fitting.parameterTransform;
            if iscell(method); method = strjoin(cellstr(method),','); end
            disp(['Transform(s)      : ', char(method)]);
            disp(['Update scheme     : ', char(fitting.updateScheme)]);
            if fitting.adaptStepSize
                disp(['Adapt step size   : true (interval ', num2str(fitting.adaptInterval), ', target ', num2str(fitting.adaptTarget), ')']);
            else
                disp( 'Adapt step size   : false');
            end
            if isfield(fitting,'adaptCovariance') && fitting.adaptCovariance
                disp( 'Adapt covariance  : true (adaptive Metropolis, burn-in only)');
            end
            disp(['Over-dispersion   : ', num2str(fitting.overdisp)]);
        end

        % Gaussian log-likelihood, same computation as metropolis_hastings
        function logL = loglik_gaussian(x_struct, y, weights, Nm, FWDfunc, varargin)
            logL = arrayfun(@logP_Gaussian, sum( weights.* (FWDfunc(x_struct,varargin{:})-y).^2, 1 ), x_struct.noise, Nm);
        end

        % forward model output must be [Nm, Nv] (measurements x voxels)
        function check_forward_size(g, Nm, Nv)
            if size(g,1) ~= Nm || size(g,2) ~= Nv || ndims(g) > 2
                error('mcmc:forwardSize', ...
                    ['mcmc: FWDfunc returned an array of size %s, but the data are [%d measurements x %d voxels]. ' ...
                     'GACELLE treats dims 1-3 of the data as spatial and dims 4+ as measurements, so single-slice ' ...
                     'data must be given as [nx, ny, 1, Nmeas], not [nx, ny, Nmeas].'], mat2str(size(g)), Nm, Nv);
            end
        end

        %% adaptation helpers
        % Robbins-Monro gain of the j-th adaptation step
        function gamma = adapt_gain(j)
            gamma = 2 * j.^(-0.6);
        end

        % one Welford step per voxel: running mean m [d,N] and sum of outer products M2 [d,d,N] (double)
        function [m, M2, n] = welford_update(m, M2, n, u)
        % u : [d,N] new state (any float class, CPU or GPU), accumulated in the class of m
            u   = cast(u, 'like', m);
            [d, N] = size(u);
            n   = n + 1;
            d1  = u - m;
            m   = m + d1 ./ n;
            d2  = u - m;
            M2  = M2 + reshape(d1, d, 1, N) .* reshape(d2, 1, d, N);
        end

        % batched lower Cholesky factor of [d,d,N] symmetric matrices (lower triangle used),
        % elementwise over pages (CPU or GPU); ok [1,N] is false where a pivot is not > 0 / not finite
        function [L, ok] = chol_batch(C)
            [d, ~, N] = size(C);
            L   = zeros(size(C), 'like', C);
            ok  = true(1, 1, N, 'like', C(1) > 0);
            for j = 1:d
                s       = C(j,j,:) - sum(L(j,1:j-1,:).^2, 2);
                ok      = ok & (s > 0) & isfinite(s);
                Ljj     = sqrt(max(s, realmin(underlyingType(s))));
                L(j,j,:) = Ljj;
                if j < d
                    L(j+1:d,j,:) = (C(j+1:d,j,:) - sum(L(j+1:d,1:j-1,:) .* L(j,1:j-1,:), 2)) ./ Ljj;
                end
            end
            ok  = reshape(ok, 1, N);
        end

        % refresh the Cholesky factors from the running moments (adaptCovariance)
        function [L, valid] = acov_refresh(M2, n, accN, Lprev, validPrev, epsRel, tinyRel)
        % M2 [d,d,N] double, n # accumulated states, accN [1,N] acceptances since the accumulation
        % start, Lprev [d,d,N] previous factor (kept where the new one is not usable)
            [d, ~, N] = size(M2);
            C       = M2 ./ max(n - 1, 1);
            C       = (C + permute(C, [2 1 3])) ./ 2;
            dC      = reshape(C, d*d, N);
            dC      = dC(1:d+1:end, :);                                 % [d,N] diagonals
            I       = eye(d, 'like', C);
            Creg    = C + epsRel .* (I .* reshape(dC, 1, d, N)) ...
                        + I .* reshape(tinyRel .* max(max(dC, [], 1), 0) + realmin('double'), 1, 1, N);
            [Lnew, ok] = mcmc.chol_batch(Creg);
            ok      = ok & all(isfinite(dC), 1) & all(dC > 0, 1) & (reshape(accN, 1, N) >= d + 1);
            L       = Lprev;
            L(:,:,ok) = cast(Lnew(:,:,ok), 'like', Lprev);
            valid   = reshape(validPrev, 1, N) | ok;
        end

        % proposal increment Lp*z per voxel: Lp [d,d,N], z [d,N] -> [d,N]
        function s = acov_step(Lp, z)
            [d, N] = size(z);
            s = reshape(sum(Lp .* reshape(z, 1, d, N), 2), d, N);
        end

        % description of the adaptive-covariance rule for out.settings
        function r = acov_rule(acWarm, acMinN, kSwitch, acScale)
            r = struct( ...
                'proposal',         'u'' = u + lambda_i L_i eps, eps ~ N(0,I_d), L_i = chol(C_i,''lower''), d = # sampled parameters', ...
                'warmup',           acWarm, ...
                'accumulation',     'Welford mean/covariance (double) of the state after every iteration, from warmup+1 to the end of burn-in', ...
                'minSamples',       acMinN, ...
                'switchIteration',  kSwitch, ...
                'refresh',          'C_i and L_i at every adaptation step from switchIteration to the end of burn-in', ...
                'regularisation',   'C <- C + 1e-6*diag(C) + 1e-12*max(diag(C))*I; keep previous L_i if chol fails or < d+1 accepted moves', ...
                'lambdaInit',       acScale, ...
                'lambdaRule',       'lambda = 2.38/sqrt(d) at the switch, then log(lambda) += 2*j^(-0.6)*(acc_j - adaptTarget)', ...
                'frozen',           'lambda_i, L_i fixed after burn-in (fixed symmetric proposal, target unchanged)');
        end

        %% parameter transforms
        % All transform helpers operate row-wise: x/u is [Nvar, ...], row k
        % uses method{k}, lb(k), ub(k). They work on CPU and GPU arrays.

        % parse transform specification into a 1xNvar cellstr (lower case)
        function method = parse_transform(spec, Nvar)
            valid = {'linear','sigmoid','log'};
            if ischar(spec) || (isstring(spec) && isscalar(spec))
                method = repmat({lower(char(spec))}, 1, Nvar);
            elseif iscell(spec) || isstring(spec)
                method = cellfun(@(s) lower(char(s)), cellstr(spec), 'UniformOutput', false);
                method = method(:).';
                if numel(method) ~= Nvar
                    error('mcmc:invalidTransform', ...
                        'mcmc: parameterTransform has %d entries but there are %d model parameters.', numel(method), Nvar);
                end
            else
                error('mcmc:invalidTransform', 'mcmc: parameterTransform must be a string or a cell of strings.');
            end
            isValid = ismember(method, valid);
            if ~all(isValid)
                error('mcmc:invalidTransform', ...
                    'mcmc: unknown parameterTransform ''%s'' (valid: %s).', method{find(~isValid,1)}, strjoin(valid,', '));
            end
        end

        % check that the bounds are compatible with the transforms
        function check_transform_bounds(method, lb, ub, modelParams)
            if nargin < 4; modelParams = arrayfun(@(k) sprintf('#%d',k), 1:numel(method), 'UniformOutput', false); end
            for k = 1:numel(method)
                if strcmp(method{k},'linear'); continue; end
                if ~(isfinite(lb(k)) && isfinite(ub(k)) && ub(k) > lb(k))
                    error('mcmc:invalidBounds', ...
                        'mcmc: parameter %s with transform ''%s'' needs finite bounds with ub > lb.', modelParams{k}, method{k});
                end
                if strcmp(method{k},'log') && lb(k) < 0
                    error('mcmc:invalidBounds', ...
                        'mcmc: parameter %s with transform ''log'' needs lb >= 0 (got %g).', modelParams{k}, lb(k));
                end
            end
        end

        % native -> u
        function u = transform_forward(x, method, lb, ub)
            Nvar    = size(x,1);
            method  = mcmc.parse_transform(method, Nvar);
            u       = x;
            for k = 1:Nvar
                switch method{k}
                    case 'sigmoid'
                        % clamp slightly inside the box, then logit
                        epsB    = 1e-4 * (ub(k) - lb(k));
                        xc      = min(max(x(k,:), lb(k)+epsB), ub(k)-epsB);
                        t       = (xc - lb(k)) ./ (ub(k) - lb(k));
                        u(k,:)  = log(t) - log1p(-t);
                    case 'log'
                        if isfinite(ub(k))
                            epsB    = 1e-4 * (ub(k) - lb(k));
                            xc      = min(max(x(k,:), lb(k)+epsB), ub(k)-epsB);
                        else
                            % unbounded above (hierarchical 'log' of mcmc_bayes, lb = 0): only keep x > 0
                            xc      = max(max(x(k,:), lb(k)), 1e-30);
                        end
                        u(k,:)  = log(xc);
                end
            end
        end

        % u -> native
        function x = transform_inverse(u, method, lb, ub)
            Nvar    = size(u,1);
            method  = mcmc.parse_transform(method, Nvar);
            x       = u;
            for k = 1:Nvar
                switch method{k}
                    case 'sigmoid'
                        % 1/(1+exp(-u)) is finite for all u (exp overflow -> 0)
                        x(k,:)  = lb(k) + (ub(k) - lb(k)) ./ (1 + exp(-u(k,:)));
                    case 'log'
                        x(k,:)  = exp(u(k,:));
                end
            end
        end

        % log|dx/du| as a function of u
        function logJ = transform_logjac(u, method, lb, ub)
            Nvar    = size(u,1);
            method  = mcmc.parse_transform(method, Nvar);
            logJ    = zeros(size(u), 'like', u);
            for k = 1:Nvar
                switch method{k}
                    case 'sigmoid'
                        % log[(ub-lb) s(u) s(-u)], log s(u) = -softplus(-u)
                        uk          = u(k,:);
                        logJ(k,:)   = log(ub(k) - lb(k)) - mcmc.softplus(uk) - mcmc.softplus(-uk);
                    case 'log'
                        logJ(k,:)   = u(k,:);
                end
            end
        end

        % numeric code of each transform for the fused GPU kernel: 0 linear, 1 sigmoid, 2 log
        function code = transform_code(method)
            code = zeros(numel(method),1);
            code(strcmp(method,'sigmoid'))  = 1;
            code(strcmp(method,'log'))      = 2;
        end

        % fused u -> native and log|dx/du| in one GPU kernel (same maths as
        % transform_inverse and transform_logjac), used inside the sampling loop
        function [x, logJ] = transform_inverse_logjac_fused(u, code, lb, ub)
        % u         : [Nvar, Nv] gpuArray
        % code      : [Nvar, 1] gpuArray, see transform_code
        % lb, ub    : [Nvar, 1] gpuArray
            [x, logJ] = arrayfun(@transform_kernel, u, code, lb, ub);
        end

        % numerically stable log(1+exp(z))
        function y = softplus(z)
            y = max(z,0) + log1p(exp(-abs(z)));
        end

        %% diagnostics
        % split-R-hat (Gelman et al., BDA3), not rank-normalised
        function R = rhat(x)
        % Input
        % -----
        % x     : samples, [Nv, Ns, Nchains]
        % Output
        % ------
        % R     : split-R-hat, [Nv, 1] (NaN with fewer than 4 samples per chain)
        %
            if size(x,2) < 4; R = nan(size(x,1), 1); return; end
            xs          = mcmc.split_chains(double(x));
            n           = size(xs,2);
            chainMean   = mean(xs,2);                       % [Nv,1,M]
            chainVar    = var(xs,0,2);                      % [Nv,1,M]
            W           = mean(chainVar,3);
            B_n         = var(chainMean,0,3);               % B/n
            varPlus     = (n-1)/n .* W + B_n;
            R           = sqrt(varPlus ./ W);
        end

        % multi-chain effective sample size (Geyer initial positive + monotone sequence, on split chains)
        function N_eff = ess(x)
        % Input
        % -----
        % x     : samples, [Nv, Ns, Nchains]
        % Output
        % ------
        % N_eff : effective sample size, [Nv, 1] (NaN with fewer than 4 samples per chain)
        %
            if size(x,2) < 4; N_eff = nan(size(x,1), 1); return; end
            xs          = mcmc.split_chains(double(x));
            [~, n, M]   = size(xs);

            chainMean   = mean(xs,2);
            xc          = xs - chainMean;
            % autocovariance per chain via FFT (biased, divided by n)
            nfft        = 2^nextpow2(2*n);
            f           = fft(xc, nfft, 2);
            acov        = real(ifft(abs(f).^2, [], 2));
            acov        = acov(:,1:n,:) ./ n;               % [Nv,n,M]

            chainVar    = acov(:,1,:) .* n ./ (n-1);        % unbiased per-chain variance
            W           = mean(chainVar,3);                 % [Nv,1]
            varPlus     = (n-1)/n .* W;
            if M > 1; varPlus = varPlus + var(chainMean,0,3); end

            rho         = 1 - (W - mean(acov,3)) ./ varPlus;   % [Nv,n]

            % Geyer: pair sums P_k = rho_2k + rho_2k+1, truncated at the first non-positive pair, made monotone
            K           = floor(n/2);
            P           = rho(:,1:2:2*K) + rho(:,2:2:2*K);  % [Nv,K]
            isPos       = cumprod(P > 0, 2);
            P           = cummin(P, 2);
            tau         = -1 + 2 .* sum(P .* isPos, 2);
            tau         = max(tau, 1/log10(n*M));           % guard as in Stan (tau >= 1/log10(N))
            N_eff       = n*M ./ tau;
            N_eff(~(varPlus > 0)) = NaN;
        end

        % split each chain in two halves (drop the first sample if Ns is odd)
        function xs = split_chains(x)
            n = size(x,2);
            if mod(n,2) == 1; x = x(:,2:end,:); n = n-1; end
            h  = n/2;
            xs = cat(3, x(:,1:h,:), x(:,h+1:end,:));
        end

        % reshape [Nv, ...] masked data to [x,y,z, ...] image (any number of trailing dims)
        function img = vec2image(v, mask)
            extraDims   = size(v,2:max(ndims(v),2));
            img         = utils.reshape_ND2image(reshape(v, size(v,1), []), mask);
            img         = reshape(img, [size(mask,1:3), extraDims]);
        end

        % attach the sampler diagnostics to the output structure (opt-in options, and mcmc_bayes)
        function out = diagnostics2out(out, xPosterior, diagnostics, mask)
        % Input
        % -----
        % out           : res2out output
        % xPosterior    : structure, native-space samples, each field [Nv, Ns, Nrep]
        % diagnostics   : sampler diagnostics (.acceptance, .acceptanceBlocks, .stepSize, .sampledParams,
        %                 optional .cacheCheck (mcmc_bayes, test only) and .adaptCovariance)
        % mask          : signal mask
        %
            fields  = fieldnames(xPosterior);
            Nrep    = size(xPosterior.(fields{1}),3);
            sampled = diagnostics.sampledParams;

            % acceptance [x,y,z,Nblock,Nrep] and final u-space step size [x,y,z,Nrep] (sampled parameters only)
            out.diagnostics.acceptance          = mcmc.vec2image(diagnostics.acceptance,mask);
            out.diagnostics.acceptanceBlocks    = diagnostics.acceptanceBlocks;
            for kvar = 1:numel(sampled)
                out.diagnostics.stepSize.(sampled{kvar}) = mcmc.vec2image(permute(diagnostics.stepSize(:,kvar,:),[1 3 2]),mask);
            end

            % R-hat (repetition > 1) and ESS on native-space samples, per voxel per parameter
            chunk = 4096;   % voxels per chunk to bound the FFT memory
            for kvar = 1:numel(fields)
                xp      = xPosterior.(fields{kvar});
                Nv      = size(xp,1);
                essV    = zeros(Nv,1,'single');
                rhatV   = zeros(Nv,1,'single');
                for kc = 1:chunk:Nv
                    idx = kc:min(kc+chunk-1,Nv);
                    essV(idx) = mcmc.ess(xp(idx,:,:));
                    if Nrep > 1; rhatV(idx) = mcmc.rhat(xp(idx,:,:)); end
                end
                out.diagnostics.ess.(fields{kvar}) = utils.reshape_ND2image(essV,mask);
                if Nrep > 1; out.diagnostics.rhat.(fields{kvar}) = utils.reshape_ND2image(rhatV,mask); end
            end

            % test-only cache check (mcmc_bayes)
            if isfield(diagnostics,'cacheCheck'); out.diagnostics.cacheCheck = diagnostics.cacheCheck; end

            % adaptive covariance: final proposal covariance [x,y,z,d,d,Nrep], lambda/condition/valid [x,y,z,Nrep]
            if isfield(diagnostics,'adaptCovariance')
                ac = diagnostics.adaptCovariance;
                out.diagnostics.adaptCovariance = struct('params', {ac.params}, ...
                    'proposalCov',  mcmc.vec2image(ac.proposalCov, mask), ...
                    'lambda',       mcmc.vec2image(ac.lambda, mask), ...
                    'condition',    mcmc.vec2image(ac.condition, mask), ...
                    'valid',        mcmc.vec2image(ac.valid, mask), ...
                    'switchIteration', ac.switchIteration);
            end
        end

    end
end

% elementwise kernel of mcmc.transform_inverse_logjac_fused (GPU arrayfun)
function [x, logJ] = transform_kernel(u, code, lb, ub)
if code == 1
    % sigmoid: x = lb + (ub-lb) s(u), log|dx/du| = log(ub-lb) - softplus(u) - softplus(-u)
    %                                             = log(ub-lb) - |u| - 2 log1p(exp(-|u|))
    e       = exp(-abs(u));
    if u >= 0; s = 1/(1+e); else; s = e/(1+e); end
    x       = lb + (ub-lb)*s;
    logJ    = log(ub-lb) - abs(u) - 2*log1p(e);
elseif code == 2
    % log: x = exp(u), log|dx/du| = u
    x       = exp(u);
    logJ    = u;
else
    % linear
    x       = u;
    logJ    = u*0;
end
end