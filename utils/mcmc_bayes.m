classdef mcmc_bayes < mcmc
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
%
% *** EXPERIMENTAL ***
% This is a subclass of mcmc for Bayesian extensions of the MCMC sampler
% (parameter transforms, marginal likelihoods, hierarchical and MRF priors).
% Interface and behaviour may change without notice.
%
% With all new options at their legacy defaults (or absent), optimisation
% passes through to mcmc.optimisation unchanged, so the output is bitwise
% identical to mcmc.
%
% New fitting options (legacy defaults)
%   .parameterTransform : 'linear'      'linear'|'sigmoid'|'log', or a cell with one entry per modelParams
%   .likelihood         : 'gaussian'    (Phase 2+, not implemented yet)
%   .S0Param            : ''            (Phase 2+, not implemented yet)
%   .updateScheme       : 'joint'       'joint'|'componentwise'
%   .adaptStepSize      : false         adapt proposal scale during burn-in (frozen afterwards)
%   .adaptInterval      : 50            # iterations between two adaptation steps
%   .adaptTarget        : []            target acceptance rate, [] -> 0.234 (joint) | 0.44 (componentwise)
%   .overdisp           : 0             relative over-dispersion of the start point for repetition > 1
%   .prior              : []            (Phase 2+, not implemented yet)
%
% Test-only option (not part of the user interface)
%   .forceNewPath       : false         run the new sampling loop even if all new options are at their
%                                       legacy defaults. Used to compare the neutral new path against
%                                       the legacy mcmc in tests/validation/mcmc_bayes.
%
% Phase 1 (sampler infrastructure) design notes
% ---------------------------------------------
% Target. Still the legacy target: Gaussian likelihood (logP_Gaussian with
%   the sampled 'noise') times a uniform box prior on [lb,ub] in native
%   space. Sampling happens in transformed space u = T(x), so the log-target
%   in u is loglik(x(u)) + sum_p log|dx_p/du_p|.
%
% Transforms (per parameter)
%   'linear'  : x = u.                          Box enforced by rejection (as legacy).
%   'sigmoid' : x = lb + (ub-lb)*sigmoid(u).    Box never violated, no rejection.
%   'log'     : x = exp(u), requires lb >= 0.   ub (and lb if lb > 0) enforced by rejection.
%   Native start points are clamped to [lb+eps, ub-eps], eps = 1e-4*(ub-lb),
%   before the forward transform of 'sigmoid'/'log' (as askadam.rescale_parameters).
%   log-sigmoid is computed via softplus for numerical stability.
%
% Proposal scale. Per voxel per parameter, in u space. It is initialised
%   from fitting.xStepSize (a native-space step) mapped through the local
%   derivative at the start point: sigma_u = xStepSize / |dx/du|(u0). For
%   'linear' this is exactly xStepSize. For non-linear transforms sigma_u
%   is capped at the u-width of the eps-clamped box.
%
% Update schemes
%   'joint'         : all parameters of a voxel proposed together (1 forward evaluation per iteration)
%   'componentwise' : one parameter at a time, sequentially within the sweep,
%                     the cached loglik is updated after each step (Nvar forward evaluations per iteration)
%   All voxels are independent and accepted in parallel.
%
% Adaptation (burn-in only, frozen afterwards). Every adaptInterval iterations
%   (as long as the window ends within the burn-in), Robbins-Monro on the log scale:
%       log sigma <- log sigma + gamma_j * (acc_j - adaptTarget),  gamma_j = 2 * j^(-0.6)
%   where acc_j is the acceptance rate of the j-th window. For 'joint', one
%   update per voxel scales all parameters together (ratios kept); for
%   'componentwise', one update per voxel per parameter.
%
% Over-dispersed starts. For repetition ii > 1 (and overdisp > 0):
%       u0_ii = u0 + overdisp * W .* randn,   W_p = T_p(ub_p) - T_p(lb_p)
%   i.e. overdisp is a fraction of the u-width of the (eps-clamped) box; the
%   result is clamped back into that box. For 'linear', W = ub-lb, the same
%   scale that mcmc uses for 'startRange'. overdisp = 0 gives the legacy
%   behaviour (all repetitions start from the same point).
%
% Diagnostics (computed on native-space samples [Nv, Ns, Nchains])
%   rhat : split-R-hat (Gelman et al., BDA3, 2013), not rank-normalised
%   ess  : multi-chain ESS on split chains with Geyer's initial positive and
%          monotone sequence estimator, as in Stan / Vehtari et al. (2021)
%          but without rank normalisation
%
% Date created: 26 September 2026
% Date modified: 26 September 2026 (Phase 1: transforms, update schemes, adaptation, overdisp, diagnostics)
%

    methods
        function out = optimisation(this, data, mask, weights, pars0, fitting, FWDfunc, varargin)
        % Input
        % ----------
        % Same as mcmc.optimisation, plus the new fitting options listed in
        % the class header.
        %

            [isLegacy, nonDefault] = mcmc_bayes.isLegacy(fitting);
            forceNewPath = isstruct(fitting) && isfield(fitting,'forceNewPath') && ~isempty(fitting.forceNewPath) && fitting.forceNewPath;

            % legacy path: identical to mcmc
            if isLegacy && ~forceNewPath
                out = optimisation@mcmc(this, data, mask, weights, pars0, fitting, FWDfunc, varargin{:});
                return
            end

            % Phase 2+ options are not available yet
            notImplemented = intersect(nonDefault, {'likelihood','S0Param','prior'}, 'stable');
            if ~isempty(notImplemented)
                error('mcmc_bayes:notImplemented', ...
                    'mcmc_bayes: non-default option(s) not implemented yet: %s', strjoin(notImplemented, ', '));
            end

            fitting = this.check_set_default_bayes(fitting);

            % only Metropolis-Hastings on the new path
            if ~strcmpi(fitting.algorithm,'mh')
                error('mcmc_bayes:unsupportedAlgorithm', ...
                    'mcmc_bayes: the new sampling path supports fitting.algorithm = ''MH'' only (got ''%s'').', fitting.algorithm);
            end

            % Step 0: display basic messages
            this.display_basic_algorithm_parameters(fitting);
            this.display_bayes_algorithm_parameters(fitting);

            % mask data to reduce memory load, same as mcmc; keep mask
            % geometry for later phases (neighbour tables for MRF priors)
            mask_idx = find(mask>0);
            geom     = struct('mask_idx', mask_idx, 'dims', size(mask));
            if ~ismatrix(data);     data    = utils.reshape_ND2GD(data,      mask_idx); else; data = data(:,mask_idx);     end
            if ~ismatrix(weights);  weights = utils.reshape_ND2GD(weights,   mask_idx); elseif ~isempty(weights); weights = weights(:,mask_idx);  end
            pars0 = utils.reshape_ND2GD_struct(pars0,mask);

            % MCMC
            [xPosterior, diagnostics] = this.metropolis_hastings_bayes(data, pars0, weights, fitting, geom, FWDfunc, varargin{:});

            % finish up
            out = this.res2out(xPosterior,fitting,mask,diagnostics);

        end

        function [xPosterior, diagnostics] = metropolis_hastings_bayes(this,y,x0,weights,fitting,geom,FWDfunc,varargin)
        % Input
        % ------
        % y         : measurements, [Nmeas,Nvoxels]
        % x0        : structure array, starting points, N fields, each field 1xNvoxel
        % weights   : weighting for non-linear least square fitting, same dimension as y
        % fitting   : Structure variable containing all fitting algorithm setting, see mcmc.metropolis_hastings
        %             and the class header for the new options
        % geom      : structure, mask geometry (.mask_idx, .dims), unused in Phase 1
        % FWDfunc   : function handle for forward signal model
        % varargin  : other input required for @FWDfunc
        %
        % Output
        % ------
        % xPosterior    : structure, native-space posterior samples, each field [Nvoxel, Nsample, Nrepetition] (as mcmc)
        % diagnostics   : structure
        %   .acceptance     : post-burn-in acceptance rate, [Nvoxel, Nblock, Nrepetition], Nblock = 1 (joint) or Nvar (componentwise)
        %   .stepSize       : final u-space proposal scale, [Nvoxel, Nvar, Nrepetition]
        %   .settings       : resolved sampler settings and RNG states
        %
            fitting = this.check_set_default_bayes(fitting);
            if isempty(weights); weights = ones(size(y), 'like', y); end

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

            % Gaussian likelihood needs the sampled noise
            if ~any(strcmp(fitting.modelParams,'noise'))
                error('mcmc_bayes:noNoise', 'mcmc_bayes: likelihood ''gaussian'' requires ''noise'' in fitting.modelParams.');
            end
            if numel(fitting.xStepSize) ~= Nvar
                error('mcmc_bayes:xStepSize', 'mcmc_bayes: fitting.xStepSize must have one entry per modelParams (%d).', Nvar);
            end

            % transforms
            method      = this.parse_transform(fitting.parameterTransform, Nvar);
            this.check_transform_bounds(method, fitting.lb, fitting.ub, fitting.modelParams);
            isLinear    = strcmp(method,'linear');
            hasJac      = ~all(isLinear);                           % false -> neutral path, skip all Jacobian terms
            isRejectRow = gpuArray(~strcmp(method(:),'sigmoid'));   % rows whose box is enforced by rejection
            code        = gpuArray(single(this.transform_code(method)));  % [Nvar,1] for the fused GPU kernel
            allReject   = ~any(strcmp(method,'sigmoid'));           % true -> same bound check as legacy

            % update scheme and adaptation
            isComponent = strcmpi(fitting.updateScheme,'componentwise');
            if isComponent; Nblock = Nvar; else; Nblock = 1; end
            isAdapt     = logical(fitting.adaptStepSize);
            Nadapt      = floor(Nburnin/fitting.adaptInterval);     % # adaptation windows, all within burn-in
            if isAdapt && Nburnin < 2*fitting.adaptInterval
                warning('mcmc_bayes:shortBurnin', ...
                    'adaptStepSize is on but Nburnin (%d) < 2*adaptInterval (%d); %d adaptation step(s) only.', ...
                    Nburnin, 2*fitting.adaptInterval, Nadapt);
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

            % log-likelihood of a native-space parameter array [Nvar,Nv]
            loglik = @(x) mcmc_bayes.loglik_gaussian(this.array2struct(x,fitting.modelParams), y, weights, Nm, FWDfunc, varargin{:});

            % starting point in native space (same as mcmc), then in u space
            xStart  = this.struct2array(x0,fitting.modelParams);      % extract parameter structure to numeric array for faster computation
            xStart  = max(xStart,lb); xStart = min(xStart,ub);         % set boundary
            uStart  = this.transform_forward(xStart, method, lb, ub);

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

            counter = 0; start = tic;
            for k = 1:fitting.iteration

                if ~isComponent
                    % ========== joint update: all parameters of a voxel together ==========
                    % 1. make a proposal with normal distribution in u space
                    uProposed       = uCurr + sigma.*randn(size(uCurr),'like',uCurr);
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
                    logLProposed    = loglik(xProposed);
                    % 2.2 accept with probability min(1, exp(logRatio)); NaN is rejected
                    if hasJac
                        logRatio            = logLProposed - logLCurr + sum(logJProposed - logJCurr, 1);
                        isAccepted          = exp(logRatio) > rand(1,Nv,'like',logLProposed);
                    else
                        % neutral path: same expression as mcmc.metropolis_hastings (the GPU evaluates
                        % fused elementwise expressions slightly differently, so keep it verbatim)
                        acceptanceRatio     = min(exp(logLProposed-logLCurr), 1);
                        isAccepted          = acceptanceRatio > rand(1,Nv,'like',logLProposed);
                        isOutofbound        = isOutofbound | isnan(logLProposed);  % mcmc would accept NaN via min(NaN,1) = 1
                    end
                    isAccepted(isOutofbound)= 0;    % reject out of bound (and NaN) proposal
                    % 2.3 update parameters if accepted
                    logLCurr(isAccepted)    = logLProposed(isAccepted);
                    uCurr(:,isAccepted)     = uProposed(:,isAccepted);
                    % neutral path (all linear): x == u, only u is tracked in the joint loop
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

                        % 2. Metropolis sampling, cached loglik is the current state
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

                % 3. acceptance bookkeeping and adaptation (burn-in only, frozen afterwards)
                if k <= Nburnin
                    if isAdapt
                        accWin = accWin + isAccepted;
                        if mod(k, fitting.adaptInterval) == 0
                            jAdapt  = jAdapt + 1;
                            delta   = this.adapt_gain(jAdapt) .* (accWin./fitting.adaptInterval - fitting.adaptTarget);
                            % joint: delta is [1,Nv] and scales all parameters of a voxel together
                            sigma   = sigma .* exp(delta);
                            if hasJac; sigma(~isLinear,:) = min(sigma(~isLinear,:), uWidth(~isLinear)); end
                            accWin(:) = 0;
                        end
                    end
                else
                    accPost = accPost + isAccepted;
                end

                % 4. Maintain the independence between iterations
                % 4.1 discard the first burnin*100% iterations
                % 4.2 keep an iteration every N iterations
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
            end

            % convert final posterior distribution into structure
            xPosterior = this.array2struct(xPosterior,fitting.modelParams);
            for kvar = 1:Nvar; xPosterior.(fitting.modelParams{kvar}) = shiftdim(xPosterior.(fitting.modelParams{kvar}),1); end

            % diagnostics and resolved settings
            diagnostics.acceptance  = acceptance;
            if isComponent; diagnostics.acceptanceBlocks = fitting.modelParams(:).'; else; diagnostics.acceptanceBlocks = {'joint'}; end
            diagnostics.stepSize    = stepSize;
            diagnostics.settings    = struct( ...
                'parameterTransform',   {method}, ...
                'updateScheme',         lower(fitting.updateScheme), ...
                'adaptStepSize',        isAdapt, ...
                'adaptInterval',        fitting.adaptInterval, ...
                'adaptTarget',          fitting.adaptTarget, ...
                'adaptRule',            'log(sigma) += 2*j^(-0.6)*(acc_j - adaptTarget), every adaptInterval iterations within burn-in', ...
                'Nadapt',               Nadapt*isAdapt, ...
                'Nburnin',              Nburnin, ...
                'overdisp',             fitting.overdisp, ...
                'overdispRule',         'u0 + overdisp*(T(ub)-T(lb)).*randn for repetition > 1', ...
                'stepSizeInit',         'xStepSize ./ |dx/du| at the start point (u space)', ...
                'forceNewPath',         logical(fitting.forceNewPath), ...
                'rngState',             rngState, ...
                'gpuRngState',          gpuRngState, ...
                'geom',                 geom);

            disp('The Metropolis-Hastings MCMC sampling (mcmc_bayes) is completed.')

        end

    end

    methods(Static)

        % check whether all new options are absent or at their legacy defaults
        function [tf, nonDefault] = isLegacy(fitting)
        % Output
        % ------
        % tf            : true if no new option is set to a non-default value
        % nonDefault    : cell array of names of the non-default option(s)
        %
            nonDefault = {};

            if isempty(fitting) || ~isstruct(fitting)
                tf = true;
                return
            end

            % option name, legacy default
            defaults = { 'parameterTransform',  'linear';
                         'likelihood',          'gaussian';
                         'S0Param',             '';
                         'updateScheme',        'joint';
                         'adaptStepSize',       false;
                         'adaptInterval',       50;
                         'adaptTarget',         [];
                         'overdisp',            0;
                         'prior',               []};

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

            tf = isempty(nonDefault);
        end

        % check and set default fitting algorithm parameters for the new path
        function fitting2 = check_set_default_bayes(fitting)
        % Input
        % -----
        % fitting       : structure contains fitting algorithm parameters, see class header
        %
            fitting2 = mcmc.check_set_default_basic(fitting);

            if ~isfield(fitting,'parameterTransform');  fitting2.parameterTransform = 'linear';      end
            if ~isfield(fitting,'likelihood');          fitting2.likelihood         = 'gaussian';    end
            if ~isfield(fitting,'S0Param');             fitting2.S0Param            = '';            end
            if ~isfield(fitting,'updateScheme');        fitting2.updateScheme       = 'joint';       end
            if ~isfield(fitting,'adaptStepSize');       fitting2.adaptStepSize      = false;         end
            if ~isfield(fitting,'adaptInterval');       fitting2.adaptInterval      = 50;            end
            if ~isfield(fitting,'adaptTarget');         fitting2.adaptTarget        = [];            end
            if ~isfield(fitting,'overdisp');            fitting2.overdisp           = 0;             end
            if ~isfield(fitting,'prior');               fitting2.prior              = [];            end
            if ~isfield(fitting,'forceNewPath');        fitting2.forceNewPath       = false;         end

            if ~any(strcmpi(fitting2.updateScheme,{'joint','componentwise'}))
                error('mcmc_bayes:invalidUpdateScheme', ...
                    'mcmc_bayes: fitting.updateScheme must be ''joint'' or ''componentwise'' (got ''%s'').', char(fitting2.updateScheme));
            end
            % target acceptance rate per update scheme
            if isempty(fitting2.adaptTarget)
                if strcmpi(fitting2.updateScheme,'componentwise'); fitting2.adaptTarget = 0.44; else; fitting2.adaptTarget = 0.234; end
            end
            if isempty(fitting2.overdisp); fitting2.overdisp = 0; end
        end

        % display the new sampler settings
        function display_bayes_algorithm_parameters(fitting)
            method = fitting.parameterTransform;
            if iscell(method); method = strjoin(cellstr(method),','); end
            disp(['Transform(s)      : ', char(method)]);
            disp(['Update scheme     : ', char(fitting.updateScheme)]);
            if fitting.adaptStepSize
                disp(['Adapt step size   : true (interval ', num2str(fitting.adaptInterval), ', target ', num2str(fitting.adaptTarget), ')']);
            else
                disp( 'Adapt step size   : false');
            end
            disp(['Over-dispersion   : ', num2str(fitting.overdisp)]);
        end

        % Gaussian log-likelihood, same computation as mcmc.metropolis_hastings
        function logL = loglik_gaussian(x_struct, y, weights, Nm, FWDfunc, varargin)
            logL = arrayfun(@logP_Gaussian, sum( weights.* (FWDfunc(x_struct,varargin{:})-y).^2, 1 ), x_struct.noise, Nm);
        end

        % Robbins-Monro gain of the j-th adaptation step
        function gamma = adapt_gain(j)
            gamma = 2 * j.^(-0.6);
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
                    error('mcmc_bayes:invalidTransform', ...
                        'mcmc_bayes: parameterTransform has %d entries but there are %d model parameters.', numel(method), Nvar);
                end
            else
                error('mcmc_bayes:invalidTransform', 'mcmc_bayes: parameterTransform must be a string or a cell of strings.');
            end
            isValid = ismember(method, valid);
            if ~all(isValid)
                error('mcmc_bayes:invalidTransform', ...
                    'mcmc_bayes: unknown parameterTransform ''%s'' (valid: %s).', method{find(~isValid,1)}, strjoin(valid,', '));
            end
        end

        % check that the bounds are compatible with the transforms
        function check_transform_bounds(method, lb, ub, modelParams)
            if nargin < 4; modelParams = arrayfun(@(k) sprintf('#%d',k), 1:numel(method), 'UniformOutput', false); end
            for k = 1:numel(method)
                if strcmp(method{k},'linear'); continue; end
                if ~(isfinite(lb(k)) && isfinite(ub(k)) && ub(k) > lb(k))
                    error('mcmc_bayes:invalidBounds', ...
                        'mcmc_bayes: parameter %s with transform ''%s'' needs finite bounds with ub > lb.', modelParams{k}, method{k});
                end
                if strcmp(method{k},'log') && lb(k) < 0
                    error('mcmc_bayes:invalidBounds', ...
                        'mcmc_bayes: parameter %s with transform ''log'' needs lb >= 0 (got %g).', modelParams{k}, lb(k));
                end
            end
        end

        % native -> u
        function u = transform_forward(x, method, lb, ub)
            Nvar    = size(x,1);
            method  = mcmc_bayes.parse_transform(method, Nvar);
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
                        epsB    = 1e-4 * (ub(k) - lb(k));
                        xc      = min(max(x(k,:), lb(k)+epsB), ub(k)-epsB);
                        u(k,:)  = log(xc);
                end
            end
        end

        % u -> native
        function x = transform_inverse(u, method, lb, ub)
            Nvar    = size(u,1);
            method  = mcmc_bayes.parse_transform(method, Nvar);
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
            method  = mcmc_bayes.parse_transform(method, Nvar);
            logJ    = zeros(size(u), 'like', u);
            for k = 1:Nvar
                switch method{k}
                    case 'sigmoid'
                        % log[(ub-lb) s(u) s(-u)], log s(u) = -softplus(-u)
                        uk          = u(k,:);
                        logJ(k,:)   = log(ub(k) - lb(k)) - mcmc_bayes.softplus(uk) - mcmc_bayes.softplus(-uk);
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
        % R     : split-R-hat, [Nv, 1]
        %
            xs          = mcmc_bayes.split_chains(double(x));
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
        % N_eff : effective sample size, [Nv, 1]
        %
            xs          = mcmc_bayes.split_chains(double(x));
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

        % convert estimation into organised output structure
        function out = res2out(xPosterior,fitting,mask,diagnostics)
        % Same as mcmc.res2out; with the 4th input (new path only), also
        % attaches out.diagnostics and out.settings. The legacy call with
        % 3 inputs is passed through unchanged.
        %
            out = res2out@mcmc(xPosterior,fitting,mask);
            if nargin < 4 || isempty(diagnostics); return; end

            fields  = fieldnames(xPosterior);
            Nrep    = size(xPosterior.(fields{1}),3);

            % acceptance [x,y,z,Nblock,Nrep] and final u-space step size [x,y,z,Nrep]
            out.diagnostics.acceptance          = mcmc_bayes.vec2image(diagnostics.acceptance,mask);
            out.diagnostics.acceptanceBlocks    = diagnostics.acceptanceBlocks;
            for kvar = 1:numel(fields)
                idx = find(strcmp(fitting.modelParams,fields{kvar}));
                out.diagnostics.stepSize.(fields{kvar}) = mcmc_bayes.vec2image(permute(diagnostics.stepSize(:,idx,:),[1 3 2]),mask);
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
                    essV(idx) = mcmc_bayes.ess(xp(idx,:,:));
                    if Nrep > 1; rhatV(idx) = mcmc_bayes.rhat(xp(idx,:,:)); end
                end
                out.diagnostics.ess.(fields{kvar}) = utils.reshape_ND2image(essV,mask);
                if Nrep > 1; out.diagnostics.rhat.(fields{kvar}) = utils.reshape_ND2image(rhatV,mask); end
            end

            out.settings = diagnostics.settings;
        end

    end

end

% elementwise kernel of mcmc_bayes.transform_inverse_logjac_fused (GPU arrayfun)
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
