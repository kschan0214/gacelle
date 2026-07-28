classdef askadam < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% 
% This is the class of all askadam realted functions
%
% Date created: 4 April 2024 
% Date modified: 21 August 2024
% Date modified: 3 October 2024
% Date modified: 11 April 2025
% Date modified: 11 June 2026 (add more convergence options)

    properties (Constant)
        epsilon = 1e-8;
    end
%
    properties (GetAccess = public, SetAccess = protected)

    end

    methods

        function [gradients,loss,loss_fidelity,loss_reg,residuals,minGPUMem] = model_gradient(this, parameters, data, mask, weights, fitting, userfuncCell, varargin)
        % Input
        % ----------
        % parameters    : Structure variable containing all parameters to be estimated
        % data          : N-D measureemnt data
        % mask          : M-D signal mask (M=[1,3])
        % weights       : N-D weights for optimisaiton
        % fitting       : Structure variable containing all fitting algorithm setting
        % userfuncCell  : cell array containing the forward model and optional regularisation function handles 
        % varargin      : contains additional input requires for FWDfunc
        % 
        % Output
        % ------
        % gradients     : Adam gradient
        % loss          : total loss
        % loss_fidelity : loss associated with data fidelity (consistancy)
        % loss_reg      : loss associated with (TV) regularisation

        if fitting.debug
            gpu        = gpuDevice;
            minGPUMem  = gpu.AvailableMemory;
        else
            minGPUMem = inf;
        end
            
            % Obatin user forwsrd model function and regularisation function
            if ~iscell(userfuncCell)
                % for legacy input format
                userfuncCell    = {userfuncCell}; 
                modelInput      = varargin;
            else
                % get extra input for the forward mdoel
                modelInput = varargin{:}{1};
            end

            FWDfunc = userfuncCell{1};
            if numel(userfuncCell) > 1
                % user defined regularisation function handle
                REGfunc = userfuncCell{2};

                % get extra input for the forward mdoel
                regulInput = varargin{:}{2};
            else
                % use default TV regularisation handle
                REGfunc = @spatial_total_variation;

                % regularisation input
                regulInput = {mask,fitting.lambda,fitting.regmap,fitting.TVmode,fitting.voxelSize};

            end

            % Forward signal simulation
            % signal_FWD = FWDfunc(this.unscale_parameters(parameters,fitting.lb,fitting.ub,fitting.modelParams),varargin{:});
            signal_FWD = FWDfunc(this.unscale_parameters(parameters,fitting.lb,fitting.ub,fitting.modelParams,fitting.parameterTransform),modelInput{:});
            % masking Forward signal if the 'signal_FWD' is not 2D
            if ~ismatrix(signal_FWD); signal_FWD = utils.reshape_ND2GD(signal_FWD, mask); end
            % ensure numerical output
            signal_FWD = utils.set_nan_inf_zero(signal_FWD);

            % vectorise
            signal_FWD = dlarray(signal_FWD(:).', 'CB');

            % Data fidelity term
            switch lower(fitting.lossFunction)
                case 'l1'
                    residuals = l1loss(signal_FWD, data, weights, Reduction="none");
                case 'l2'
                    residuals = l2loss(signal_FWD, data, weights, Reduction="none");
                case 'huber'
                    residuals = huber(signal_FWD, data, weights, Reduction="none");
                case 'mse'
                    residuals = mse(signal_FWD, data, Reduction="none");
            end
            loss_fidelity = mean(residuals);

            % regularisation term
            loss_reg = REGfunc(this.unscale_parameters(parameters,fitting.lb,fitting.ub,fitting.modelParams,fitting.parameterTransform),regulInput{:});
            
            % compute loss
            loss = loss_fidelity + loss_reg;
            
            % Calculate gradients with respect to the learnable parameters.
            gradients = dlgradient(loss,parameters);

            if ~fitting.enableComplex
                fieldname = fieldnames(gradients);
                for k = 1:numel(fieldname)
                    gradients.(fieldname{k}) = real(gradients.(fieldname{k}));
                end
            end

            if fitting.debug
                minGPUMem  = min(gpu.AvailableMemory,minGPUMem);
            end
        end

        % askAdam optimisation loop
        % function out = optimisation(this, data, mask, weights, parameters, fitting, FWDfunc, varargin)
        function out = optimisation(this, data, mask, weights, parameters, fitting, userfuncCell, varargin)
        % Input
        % -----
        % data                  : 2-D (vectorised imaging) data
        % mask                  : (1-3)D signal mask
        % weights               : N-D wieghts, same dimension as 'data' (optional)
        % parameters            : structure variable containing starting points of all model parameters to be estimated (optional)
        % fitting               : structure contains fitting algorithm parameters
        %   .modelParams        : 1xM cell variable,    name of the model parameters, e.g. {'S0','R2star'};
        %   .lb                 : 1xM numeric variable, fitting lower bound, same order as field 'modelParams', e.g. [0.5, 0];
        %   .ub                 : 1xM numeric variable, fitting upper bound, same order as field 'modelParams', e.g. [2, 1];
        %   .isDisplay          : boolean, display optimisation process in graphic plot
        %   .convergenceValue   : tolerance in loss gradient to stop the optimisation
        %   .convergenceWindow  : # of elements in which 'convergenceValue' is computed
        %   .iteration          : maximum # of optimisation iterations
        %   .initialLearnRate   : initial learn rate of Adam optimiser
        %   .tol                : tolerance in loss
        %   .lambda             : regularisation parameter(s)
        %   .regmap             : model parameter(s) in which regularisation is applied
        %   .lossFunction       : loss function, 'L1'|'L2'|'huber'|'mse'
        % userfuncCell          : cell array containing the forward model and optional regularisation function handles 
        % varargin              : additional input for FWDfunc other than 'parameter' and 'mask' (same order as FWDfunc)
        %
        % Output
        % ------
        % out                   : structure contains optimisation result
        %

            gpu             = gpuDevice;
            initFreeGPUMem  = gpu.AvailableMemory;

            dims = size(mask,1:3); mask_idx = find(mask>0);

            %%%%%%%%%%%%%%%%%%%%%%%%%% 1. I/O Setup %%%%%%%%%%%%%%%%%%%%%%%%%%
            % data can be either 2D or ND, if ND then convert to 2D here
            % masking if the input data are not 2D
            if ~ismatrix(data);     data    = utils.reshape_ND2GD(data,      mask_idx); else; data = data(:,mask_idx);     end
            if ~ismatrix(weights);  weights = utils.reshape_ND2GD(weights,   mask_idx); elseif ~isempty(weights); weights = weights(:,mask_idx);  end
                
            % the first dimension must be 'measurement' and second dimension 'voxel'
            [Nmeas,Nvol] = size(data);

            % put data into gpuArray
            mask    = gpuArray(logical(mask)); 
            data    = gpuArray(single(data)); 
            if ~isempty(weights); weights = gpuArray(single(weights)); else; weights = ones(size(data),'like',data); end
            % vectorise input data
            data    = dlarray(data(:).',    'CB');
            weights = dlarray(weights(:).', 'CB');

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%%%%%%%% 2. Initialisation %%%%%%%%%%%%%%%%%%%%%%%%%%

            % --- 2.1 Fitting settings ---
            fitting = this.check_set_default_basic(fitting);
            if numel(userfuncCell) > 1; fitting.defaultRegularisation = false; end

            % --- 2.2 Parameters ---
            % linear mode: normalise to [0,1] using parameter bounds, then clamp
            % sigmoid mode: initialise as unconstrained z values (logit of rescaled theta0); no clamping
            parameters = this.initialise_parameter(dims, parameters, fitting, mask);
            if strcmp(fitting.parameterTransform, 'linear')
                parameters = this.set_boundary01(parameters, fitting.enableComplex);
            end

            % --- 2.3 Optimiser state ---
            averageGrad   = []; averageSqGrad = []; vel           = [];

            % --- 2.4 Accelerated function handle ---
            if fitting.debug;   accfun = @this.model_gradient ;
            else;               accfun = dlaccelerate(@this.model_gradient); clearCache(accfun); end

            % --- 2.5 Starting point evaluation ---
            % compute loss and residuals at initialisation for minLoss tracking and
            % per-voxel loss history (robustConvergence)
            [~, loss, loss_fidelity, loss_reg, residuals, minGPUMem] = dlfeval(accfun, parameters, data, mask, weights, fitting, userfuncCell, varargin{:});
            
            loss                  = double(utils.dlarray2single(loss));
            perVoxelLossInit      = extractdata(mean(reshape(residuals, Nmeas, Nvol), 1));  % [1 x Nvol], CPU

            % --- 2.6 Minimum loss tracking ---
            minLoss               = loss;
            minLossFidelity       = utils.dlarray2single(loss_fidelity);
            minLossRegularisation = utils.dlarray2single(loss_reg);
            minResiduals          = residuals;
            parameters_minLoss    = parameters;
            minIteration          = 0;
            epoch                 = 0;

            % --- 2.7 Convergence signal state ---
            ema_loss                     = loss;
            convergenceBuffer            = ones(fitting.convergenceWindow, 1);
            epochsWithoutImprovementConv = 0;
            stepNorm_curr                = Inf;
            epochsWithoutImprovementStep = 0;
            parameters_prev              = parameters;
            gradNorm_curr                = Inf;
            epochsWithoutImprovementGrad = 0;

            % Convergence-mask freeze state (robustConvergence path only).
            % mainMask_conv is the voxel set the convergence signal is computed over.
            % It is refreshed only on convergenceWindow boundaries (not every
            % weightUpdateInterval), so no linear-slope/EMA window ever straddles a
            % membership change. convReprimeCount counts down the epochs after a
            % refresh during which the buffer is still repopulating with the new
            % set; convergence cannot fire while it is > 0.
            mainMask_conv                = true(1, Nvol);   % [1 x Nvol], CPU
            convReprimeCount             = 0;
            
            % --- 2.8 Robust convergence state ---
            % mainMask, outlierFlagCount, perVoxelLossHistory always initialised
            % so update_outlier_mask can be called unconditionally in the loop
            mainMask            = true(1, Nvol);                    % [1 x Nvol], CPU
            outlierFlagCount    = zeros(1, Nvol, 'single');         % [1 x Nvol], CPU
            perVoxelLossHistory = [];                               % populated below if robustConvergence
            weights_original    = weights;                          % always stored; used only if robustConvergence
            
            if fitting.robustConvergence
                perVoxelLossHistory = repmat(perVoxelLossInit, fitting.outlierCheckWindow, 1);  % [outlierCheckWindow x Nvol], CPU
            end
            % --- 2.9 GPU memory usage at initialisation ---
            currFreeGPUMem  = min(gpu.AvailableMemory, minGPUMem);
            currGPUMemUsage = (initFreeGPUMem - currFreeGPUMem) / 1024^3;
            
            % --- 2.10 Display ---
            if fitting.isDisplay; lineLoss = this.setup_display; end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%% 3. Optimisation %%%%%%%%%%%%%%%%%%%%%%%%%%
            start = tic;
            if fitting.iteration > 0
                
                % display optimisation algorithm parameters
                this.display_basic_fitting_parameters(fitting);

                disp('Optimisation begins...');
                disp('----------------------');

                for epoch = 1:fitting.iteration

                    %%%%%%%%%%%%%%%%%%%% 3.1. Forward pass %%%%%%%%%%%%%%%%%%%%
                    [gradients, loss, loss_fidelity, loss_reg, residuals] = this.forward_pass(accfun, parameters, data, mask, weights, fitting, userfuncCell, varargin{:});

                    %%%%%%%%%%%%%%%%%%%% 3.2. Track minimum loss %%%%%%%%%%%%%%%%%%%%
                    [minLoss, minLossFidelity, minLossRegularisation, minResiduals, parameters_minLoss, minIteration] = this.update_min_loss(loss, loss_fidelity, loss_reg, residuals, parameters, ...
                                                                                                                            minLoss, minLossFidelity, minLossRegularisation, minResiduals, parameters_minLoss, minIteration, epoch);

                    %%%%%%%%%%%%%%%%%%%% 3.3. Outlier detection and weight update %%%%%%%%%%%%%%%%%%%%
                    [mainMask, weights, outlierFlagCount, perVoxelLossHistory, perVoxelLoss] = this.update_outlier_mask(residuals, perVoxelLossInit, mainMask, outlierFlagCount, ...
                                                                                                                            perVoxelLossHistory, weights_original, Nmeas, Nvol, epoch, fitting);
                    % %%%%%%%%%%%%%%%%%%%% DIAGNOSTIC: robustConvergence mainMask tracking %%%%%%%%%%%%%%%%%%%%
                    % if fitting.robustConvergence
                    %     fprintf('DIAG epoch %4d | mainMask %5d/%5d (%5.1f%%) | mean loss (main) = %.6e | mean loss (full) = %.6e | full-population loss = %.6e\n', ...
                    %         epoch, sum(mainMask), Nvol, 100*sum(mainMask)/Nvol, ...
                    %         mean(perVoxelLoss(mainMask)), mean(perVoxelLoss), loss);
                    % end
                    %%%%%%%%%%%%%%%%%%%% 3.4. Convergence signals %%%%%%%%%%%%%%%%%%%%
                    [convergenceCurr, ema_loss, convergenceBuffer, epochsWithoutImprovementConv, epochsWithoutImprovementStep, stepNorm_curr, parameters_prev, ...
                         epochsWithoutImprovementGrad, gradNorm_curr, mainMask_conv, convReprimeCount] = this.update_convergence_signals(loss, mainMask, mainMask_conv, convReprimeCount, perVoxelLoss, ...
                                                                                                            parameters, parameters_prev, gradients, ema_loss, convergenceBuffer, ...
                                                                                                            epochsWithoutImprovementConv, epochsWithoutImprovementStep, stepNorm_curr, ...
                                                                                                            epochsWithoutImprovementGrad, gradNorm_curr, minLoss, epoch, fitting);
                
                    %%%%%%%%%%%%%%%%%%%% 3.5. Stopping check %%%%%%%%%%%%%%%%%%%%
                    [doStop, stopMsg] = this.check_stopping(loss, epochsWithoutImprovementConv, epochsWithoutImprovementStep, epochsWithoutImprovementGrad, fitting);
                    if doStop; fprintf(stopMsg); break; end
                
                    %%%%%%%%%%%%%%%%%%%% 3.6. Parameter update %%%%%%%%%%%%%%%%%%%%
                    [parameters, averageGrad, averageSqGrad, vel, learningRate] = ...
                        this.update_parameters(parameters, gradients, averageGrad, averageSqGrad, vel, epoch, fitting);
                
                    %%%%%%%%%%%%%%%%%%%% 3.7. Verbose output %%%%%%%%%%%%%%%%%%%%
                    if fitting.isDisplay; this.add_point_to_display(lineLoss,epoch,loss,start); end     % plot loss 
                    this.print_verbose(epoch, loss, loss_fidelity, loss_reg, learningRate, convergenceCurr, epochsWithoutImprovementConv, mainMask, Nvol, ...
                                            stepNorm_curr, epochsWithoutImprovementStep, gradNorm_curr, epochsWithoutImprovementGrad, fitting, start);

                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%%%%%%%% 4. Finalisation %%%%%%%%%%%%%%%%%%%%%%%%%%
            % display final message
            if fitting.iteration > 0
                D   = duration(0, 0, toc(start), 'Format', 'hh:mm:ss');
                fprintf('Final loss         =  %e\n',double(loss));
                fprintf('Final convergence  =  %e\n',double(convergenceCurr));
                fprintf('Final #iterations  =  %d\n',epoch);
                fprintf('Total Elapsed time =  %s\n', string(D));
            end
            % make sure the final results stay within boundary (linear mode only;
            % sigmoid mode guarantees bounds via sigmoid recovery so clamping is not needed)
            if strcmp(fitting.parameterTransform, 'linear')
                parameters         = this.set_boundary01(parameters,fitting.enableComplex);
                parameters_minLoss = this.set_boundary01(parameters_minLoss,fitting.enableComplex);
            end

            if fitting.isOptimiseMemory
                parameters          = utils.undo_masking_ND2GD_preserve_struct(parameters,mask);
                parameters_minLoss  = utils.undo_masking_ND2GD_preserve_struct(parameters_minLoss,mask);
            end
            
            % rescale the network parameters back to physical units
            parameters          = this.unscale_parameters(parameters,           fitting.lb,fitting.ub,fitting.modelParams,fitting.parameterTransform);
            parameters_minLoss  = this.unscale_parameters(parameters_minLoss,   fitting.lb,fitting.ub,fitting.modelParams,fitting.parameterTransform);
            for k = 1:numel(fitting.modelParams)
                % final iteration result
                % if ~isscalar(parameters.(fitting.modelParams{k}))
                if isequal(size(parameters.(fitting.modelParams{k}),1:3),size(mask))
                    if fitting.isMaskedOut
                        tmp = utils.dlarray2single(parameters.(fitting.modelParams{k}) .* mask); 
                        % minimum loss result
                        tmp2 = utils.dlarray2single(parameters_minLoss.(fitting.modelParams{k}) .* mask); 
                    else
                        tmp = utils.dlarray2single(parameters.(fitting.modelParams{k})); 
                        % minimum loss result
                        tmp2 = utils.dlarray2single(parameters_minLoss.(fitting.modelParams{k})); 
                    end
                else
                    tmp = utils.dlarray2single(parameters.(fitting.modelParams{k}));
                    % minimum loss result
                    tmp2 = utils.dlarray2single(parameters_minLoss.(fitting.modelParams{k}));
                end
                out.final.(fitting.modelParams{k}) = tmp;
                out.min.(fitting.modelParams{k}) = tmp2;
            end
            out.final.loss          = loss;
            out.final.loss_fidelity = utils.dlarray2single(loss_fidelity);
            out.final.loss_reg      = utils.dlarray2single(loss_reg);
            out.final.resloss       = utils.reshape_ND2image( utils.dlarray2single( mean(reshape(residuals,Nmeas,Nvol),1)).',mask);
            out.final.residual      = utils.dlarray2single( reshape(residuals,Nmeas,Nvol));
            out.final.Niteration    = epoch;

            out.min.loss            = minLoss;
            out.min.loss_fidelity   = utils.dlarray2single(minLossFidelity);
            out.min.loss_reg        = utils.dlarray2single(minLossRegularisation);
            out.min.resloss         = utils.reshape_ND2image( utils.dlarray2single( mean(reshape(minResiduals,Nmeas,Nvol),1)).',mask);
            out.min.residual        = utils.dlarray2single( reshape(minResiduals,Nmeas,Nvol));
            out.min.Niteration      = minIteration;

            out.final.memoryUsage   = currGPUMemUsage;
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        end

        % initialise network parameters
        function parameters = initialise_parameter(this,img_size,pars0,fitting,mask)
            
            % get relevant parameters
            randomness      = fitting.randomness;
            modelParams     = fitting.modelParams;
            ub              = fitting.ub;
            lb              = fitting.lb;

            for k = 1:numel(modelParams)
               
                % if starting points are provided
                if ~isempty(pars0)
                    if strcmp(fitting.parameterTransform, 'sigmoid')
                        % Logit initialisation: map physical theta0 -> z in (-inf,+inf)
                        % Clamp away from exact bounds before logit to avoid -Inf/+Inf
                        eps_bound      = 1e-4 * (ub(k) - lb(k));
                        theta0_clamped = max(min(single(pars0.(modelParams{k})), ub(k) - eps_bound), lb(k) + eps_bound);
                        tmp_norm       = askadam.rescale01(theta0_clamped, lb(k), ub(k));   % maps to (0,1)
                        tmp_z0         = log(tmp_norm ./ (1 - tmp_norm));                   % logit: (0,1) -> (-inf,+inf)
                        % random perturbation in z-space is Gaussian (not uniform) since z is unbounded
                        tmp            = (1 - randomness) * tmp_z0 + randomness * randn(size(tmp_z0), 'single');
                    else
                        % linear mode: existing rescale01 behaviour
                        tmp =   rand(size(pars0.(modelParams{k})),'single') ;     % values between [0,1]
                        tmp =  (1-randomness)* this.rescale01(pars0.(modelParams{k}), lb(k), ub(k)) + randomness*tmp;     % values between [0,1]
                    end
                else
                    if strcmp(fitting.parameterTransform, 'sigmoid')
                        % No starting point: initialise z=0 (midpoint of parameter range) + Gaussian noise
                        tmp = randomness * randn(img_size, 'single');
                    else
                        % random initialisation
                        tmp = rand(img_size,'single') ;     % values between [0,1]
                    end
                end
                % put it into dlarray
                parameters.(modelParams{k}) = gpuArray( dlarray( tmp ));
            end

            % masking if optimise memory usgae
            if fitting.isOptimiseMemory
                parameters = utils.masking_ND2GD_preserve_struct(parameters,mask); 
            end
            
        end
   
    end

    methods (Access = private)

        function [gradients, loss, loss_fidelity, loss_reg, residuals] = forward_pass(this, accfun, parameters, data, mask, weights, fitting, userfuncCell, varargin)
            % Evaluate forward model, compute loss and gradients via autodiff.
            % In linear mode, parameters are clipped to [0,1] before evaluation to enforce bounds.
            % In sigmoid mode, parameters are unconstrained z-values; clamping is skipped
            % because the sigmoid in unscale_parameters guarantees physical bounds are satisfied.

            if strcmp(fitting.parameterTransform, 'linear')
                parameters = this.set_boundary01(parameters, fitting.enableComplex);
            end

            [gradients, loss, loss_fidelity, loss_reg, residuals] = dlfeval(accfun, parameters, data, mask, weights, fitting, userfuncCell, varargin{:});

            loss = double(utils.dlarray2single(loss));
        end

        function [minLoss, minLossFidelity, minLossRegularisation, minResiduals, parameters_minLoss, minIteration] = update_min_loss(~, loss, loss_fidelity, loss_reg, residuals, parameters, ...
                                                                                                                                        minLoss, minLossFidelity, minLossRegularisation, minResiduals, parameters_minLoss, minIteration, epoch)
            % Track the parameter set corresponding to the minimum loss seen so far.
            % This guards against oscillation near convergence — the minimum loss
            % result is returned alongside the final iteration result.
            if minLoss > loss
                minLoss               = loss;
                minLossFidelity       = loss_fidelity;
                minLossRegularisation = loss_reg;
                minResiduals          = residuals;
                parameters_minLoss    = parameters;
                minIteration          = epoch;
            end
        end

        function [mainMask, weights, outlierFlagCount, perVoxelLossHistory, perVoxelLoss] = update_outlier_mask(~, residuals, perVoxelLossInit, mainMask, outlierFlagCount, ...
                                                                                                                    perVoxelLossHistory, weights_original, Nmeas, Nvol, epoch, fitting)
            % Detect outlier voxels based on improvement behaviour over time and
            % downweight their gradient contribution. Only active when
            % fitting.robustConvergence = true.
            %
            % Criterion A: voxel has not improved by outlierVoxelThres over last
            %              outlierCheckWindow checks, while median has improved by outlierPopThres
            % Criterion B: voxel has not improved by outlierInitThres from initialisation,
            %              while median has improved by outlierInitPopThres
            %
            % A voxel must fail both criteria to be flagged. Once flagged it remains
            % downweighted for at least outlierMinFlagDuration checks before reassessment.
            % Outlier classification lags by one weightUpdateInterval — this is intentional
            % since extractdata breaks the autodiff graph.
        
            % pass through defaults — returned unchanged if robustConvergence is false
            % or if weightUpdateInterval has not been reached
            % weights is reassigned below only when outlier mask is updated
            weights         = weights_original;
            perVoxelLoss    = [];              % default: empty when robustConvergence is false

            if ~fitting.robustConvergence
                return
            end
        
            perVoxelLoss = extractdata(mean(reshape(residuals, Nmeas, Nvol), 1));
        
            if mod(epoch, fitting.weightUpdateInterval) == 0
        
                % shift rolling history window: drop oldest, append current
                perVoxelLossHistory = [perVoxelLossHistory(2:end, :); perVoxelLoss];
        
                % criterion A: stagnation relative to population over last N checks
                voxelImprovementA  = (perVoxelLossHistory(1,:) - perVoxelLossHistory(end,:)) ./ (perVoxelLossHistory(1,:) + 1e-8);
                medianImprovementA = median(voxelImprovementA);
                failA = voxelImprovementA < fitting.outlierVoxelThres & ...
                        medianImprovementA  > fitting.outlierPopThres;
        
                % criterion B: stagnation relative to own initialisation loss
                voxelImprovementB  = (perVoxelLossInit - perVoxelLoss) ./ (perVoxelLossInit + 1e-8);
                medianImprovementB = median(voxelImprovementB);
                failB = voxelImprovementB < fitting.outlierInitThres & ...
                        medianImprovementB  > fitting.outlierInitPopThres;
        
                % must fail both criteria to be classified as outlier
                newOutliers = failA & failB;
        
                % increment counter for flagged voxels, decrement for recovered ones
                outlierFlagCount(newOutliers)  = outlierFlagCount(newOutliers) + 1;
                outlierFlagCount(~newOutliers) = max(outlierFlagCount(~newOutliers) - 1, 0);
        
                % voxel remains flagged until counter drops to zero (minimum flag duration)
                mainMask = outlierFlagCount == 0;
        
                % modulate original user-defined weights multiplicatively
                % preserves relative SNR weighting within both populations
                outlierModulator            = ones(1, Nvol, 'single');
                outlierModulator(~mainMask) = fitting.outlierWeight;
                outlierModulator            = repmat(outlierModulator, Nmeas, 1);
                weights = weights_original .* dlarray(gpuArray(outlierModulator(:).'), 'CB');
            end
        end

        function [convergenceCurr, ema_loss, convergenceBuffer, epochsWithoutImprovementConv, epochsWithoutImprovementStep, stepNorm_curr, parameters_prev, epochsWithoutImprovementGrad, gradNorm_curr, mainMask_conv, convReprimeCount] = update_convergence_signals(this, loss, mainMask, mainMask_conv, convReprimeCount, perVoxelLoss, ...
                                                                                                                                                                                                                                    parameters, parameters_prev, gradients, ema_loss, convergenceBuffer, ...
                                                                                                                                                                                                                                    epochsWithoutImprovementConv, epochsWithoutImprovementStep, stepNorm_curr, ...
                                                                                                                                                                                                                                    epochsWithoutImprovementGrad, gradNorm_curr, minLoss, epoch, fitting)
            % Compute all active convergence signals and update their patience counters.
            %
            % Signal 1 (always active): loss-based convergence via linear slope or EMA.
            %   robustConvergence = false : computed on the full-population loss (original,
            %       unchanged behaviour).
            %   robustConvergence = true  : computed on the main (non-outlier) population,
            %       so that a small number of persistently poorly-fitting voxels do not
            %       drive the stopping decision for the well-behaved majority.
            % Signal 2 (optional): relative parameter step norm, analogous to StepTolerance
            %   in lsqnonlin. Active when fitting.convergenceStepTol > 0.
            % Signal 3 (optional): raw gradient norm before Adam correction.
            %   Active when fitting.convergenceGradTol > 0.
        
            % --- signal 1: loss-based ---
            if ~fitting.robustConvergence
                % Original behaviour: full-population loss, untouched.
                loss_convergence = loss;
            else
                % Robust behaviour (mask-freeze scheme).
                %
                % The convergence signal must exclude outlier voxels, but the outlier
                % set (mainMask) is refreshed every weightUpdateInterval epochs by
                % update_outlier_mask. Feeding a loss computed over a set that changes
                % mid-window into the windowed linear-slope/EMA check injects level-shift
                % discontinuities that can corrupt (including sign-flip) the fitted slope
                % and trigger false-positive convergence (observed 2026-07-10).
                %
                % Fix: hold the convergence voxel set (mainMask_conv) fixed and only
                % refresh it on convergenceWindow boundaries when the outlier set has
                % actually changed, so no window ever straddles a membership change. On
                % each refresh the loss level shifts (different set), so the slope buffer
                % is repriming for the next convergenceWindow epochs, during which
                % convergence is not allowed to fire (convReprimeCount).
                %
                % Outlier voxels are still updated every iteration (downweighted, not
                % frozen) via update_outlier_mask; they are only excluded from the
                % convergence *decision*, not from optimisation.
                %
                % The "actually changed" guard matters: refreshing unconditionally every
                % convergenceWindow epochs would re-arm convReprimeCount before it clears,
                % gating the convergence test permanently. When mainMask is stable (the
                % common case once outliers settle), there is no level discontinuity to
                % protect against, so mainMask_conv is left untouched and the slope/EMA
                % window runs normally.
                % Known limitation: if the outlier set never settles (voxels oscillating
                % in and out of the flagged set at every boundary), the reprime gate can
                % stay armed and convergence will only stop on max iterations. Add a
                % change-magnitude tolerance here if that failure mode is observed.
                if mod(epoch, fitting.convergenceWindow) == 0 && ~isequal(mainMask, mainMask_conv)
                    mainMask_conv    = mainMask;                    % adopt current outlier set
                    convReprimeCount = fitting.convergenceWindow;   % buffer must refill for new set
                end
                loss_convergence = mean(perVoxelLoss(mainMask_conv));
            end
        
            switch fitting.convergenceModel
                case 'ema'
                    [ema_loss, convergenceCurr]             = this.update_convergence_ema(loss_convergence, ema_loss, fitting.emaDecay);
                case 'linear'
                    [convergenceCurr, convergenceBuffer]    = this.update_convergence([convergenceBuffer(2:end); loss_convergence]);
            end
        
            % While the convergence buffer is repriming after a mask refresh, the
            % windowed slope/EMA spans a level discontinuity and is not trustworthy;
            % hold the patience counter at zero until the window is clean again.
            if convReprimeCount > 0
                convReprimeCount             = convReprimeCount - 1;
                epochsWithoutImprovementConv = 0;
            elseif convergenceCurr > fitting.convergenceValue || epoch <= fitting.convergenceWindow
                epochsWithoutImprovementConv = 0;
            elseif ~fitting.robustConvergence && (minLoss - loss) > fitting.convergenceValue
                % Full-population minLoss override. Retained unchanged for the
                % non-robust path. Deliberately NOT applied in the robust path: it
                % uses the full loss (outliers included) and would reintroduce the
                % outlier coupling that the main-population convergence signal exists
                % to remove. In the robust path the frozen main-population slope/EMA
                % above already carries the "still improving?" signal.
                epochsWithoutImprovementConv = 0;
            else
                epochsWithoutImprovementConv = epochsWithoutImprovementConv + 1;
            end
        
            % --- signal 2: step norm ---
            if fitting.convergenceStepTol > 0
                stepNorm_num = 0;
                stepNorm_den = 0;
                for k = 1:numel(fitting.modelParams)
                    delta        = parameters.(fitting.modelParams{k}) - parameters_prev.(fitting.modelParams{k});
                    stepNorm_num = stepNorm_num + gather(sum(abs(delta(:)).^2, 'all'));
                    stepNorm_den = stepNorm_den + gather(sum(abs(parameters.(fitting.modelParams{k})(:)).^2, 'all'));
                end
                stepNorm_curr   = sqrt(stepNorm_num) / (1 + sqrt(stepNorm_den));
                parameters_prev = parameters;
        
                if stepNorm_curr < fitting.convergenceStepTol
                    epochsWithoutImprovementStep = epochsWithoutImprovementStep + 1;
                else
                    epochsWithoutImprovementStep = 0;
                end
            end
        
            % --- signal 3: gradient norm ---
            if fitting.convergenceGradTol > 0
                gradNorm_curr = 0;
                fields        = fieldnames(gradients);
                for k = 1:numel(fields)
                    gradNorm_curr = gradNorm_curr + gather(sum(abs(gradients.(fields{k})(:)).^2, 'all'));
                end
                gradNorm_curr = sqrt(gradNorm_curr);
        
                if gradNorm_curr < fitting.convergenceGradTol
                    epochsWithoutImprovementGrad = epochsWithoutImprovementGrad + 1;
                else
                    epochsWithoutImprovementGrad = 0;
                end
            end
        end

        function [doStop, stopMsg] = check_stopping(~, loss, epochsWithoutImprovementConv, epochsWithoutImprovementStep, epochsWithoutImprovementGrad, fitting)
            % Check all stopping criteria and return a flag and message.
            % Criteria are checked in order: loss convergence, loss tolerance,
            % step norm, gradient norm.
            doStop  = false;
            stopMsg = '';
        
            % loss convergence
            if epochsWithoutImprovementConv > fitting.patienceConvergence
                doStop = true;
                stopMsg = sprintf('Optimisation is done. Loss convergence below tolerance %e (patience %d).\n', ...
                    fitting.convergenceValue, fitting.patienceConvergence);
                return
            end
        
            % loss tolerance
            if loss < fitting.tol
                doStop  = true;
                stopMsg = sprintf('Optimisation is done. Loss is less than the tolerance %e.\n', fitting.tol);
                return
            end
        
            % step norm
            if fitting.convergenceStepTol > 0 && epochsWithoutImprovementStep > fitting.patienceStep
                doStop  = true;
                stopMsg = sprintf('Optimisation is done. Step norm below tolerance %e (patience %d).\n', ...
                    fitting.convergenceStepTol, fitting.patienceStep);
                return
            end
        
            % gradient norm
            if fitting.convergenceGradTol > 0 && epochsWithoutImprovementGrad > fitting.patienceGrad
                doStop  = true;
                stopMsg = sprintf('Optimisation is done. Gradient norm below tolerance %e (patience %d).\n', ...
                    fitting.convergenceGradTol, fitting.patienceGrad);
                return
            end
        end

        function [parameters, averageGrad, averageSqGrad, vel, learningRate] = update_parameters(this, parameters, gradients, averageGrad, averageSqGrad, vel, epoch, fitting)
            
            learningRate = this.update_learn_rate(fitting.initialLearnRate, fitting.decayRate, epoch);
            
            if epoch < fitting.iteration
                switch lower(fitting.optimiser)
                    case 'adam'
                        [parameters, averageGrad, averageSqGrad] = adamupdate(parameters, gradients, ...
                            averageGrad, averageSqGrad, epoch, learningRate, ...
                            fitting.adamupdateGradDecay, fitting.adamupdateSqGradDecay, fitting.adamupdateEpsilon);
                    case 'sgdm'
                        [parameters, vel] = sgdmupdate(parameters, gradients, vel, ...
                            learningRate, fitting.sgdmupdateMomentum);
                    case 'rmsprop'
                        [parameters, averageSqGrad] = rmspropupdate(parameters, gradients, averageSqGrad, ...
                            learningRate, fitting.rmspropupdateSqGradDecay, fitting.rmspropupdateEpsilon);
                end
            end
        end

        function print_verbose(~, epoch, loss, loss_fidelity, loss_reg, learningRate, ...
                                convergenceCurr, epochsWithoutImprovementConv, mainMask, Nvol, ...
                                stepNorm_curr, epochsWithoutImprovementStep, ...
                                gradNorm_curr, epochsWithoutImprovementGrad, fitting, start)
            % Print iteration status. Reported fields depend on which features are active.
            if mod(epoch, 100) ~= 0 && epoch ~= 1; return; end

            % extract from dlarray if needed
            loss_fidelity = utils.dlarray2single(loss_fidelity);
            loss_reg      = utils.dlarray2single(loss_reg);
        
            D   = duration(0, 0, toc(start), 'Format', 'hh:mm:ss');
            msg = sprintf('Iteration #%4d | Loss = %.3e (fidelity = %.3e, reg = %.3e) | LR = %.3e', ...
                epoch, loss, loss_fidelity, loss_reg, learningRate);
        
            msg = [msg sprintf(' | Conv = %.3e [patience %d/%d]', ...
                convergenceCurr, epochsWithoutImprovementConv, fitting.patienceConvergence)];
            if fitting.robustConvergence
                msg = [msg sprintf(' | Outliers = %d/%d (downweighted, gradient only)', sum(~mainMask), Nvol)];
            end
        
            if fitting.convergenceStepTol > 0
                msg = [msg sprintf(' | Step = %.3e [patience %d/%d]', ...
                    stepNorm_curr, epochsWithoutImprovementStep, fitting.patienceStep)];
            end
        
            if fitting.convergenceGradTol > 0
                msg = [msg sprintf(' | Grad = %.3e [patience %d/%d]', ...
                    gradNorm_curr, epochsWithoutImprovementGrad, fitting.patienceGrad)];
            end
        
            msg = [msg sprintf(' | Elapsed: %s\n', string(D))];
            fprintf(msg);
        end
    
    end

    methods(Static)

        %% misc.

        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default_basic(fitting)
        % Input
        % -----
        % fitting               : structure contains fitting algorithm parameters
        %   .iteration          : no. of maximum iterations, default = 4000
        %   .initialLearnRate   : initial gradient step size, defaulr = 0.01
        %   .decayRate          : decay rate of gradient step size; learningRate = initialLearnRate / (1+decayRate*epoch), default = 0.0005
        %   .convergenceValue   : convergence tolerance, based on the slope of last 'convergenceWindow' data points on loss, default = 1e-8
        %   .convergenceWindow  : number of data points to check convergence, default = 20
        %   .tol                : stop criteria on metric value, default = 1e-3
        %   .lambda             : regularisation parameter, default = 0 (no regularisation)
        %   .TVmode             : mode for TV regulariation, '2D'|'3D', default = '2D'
        %   .regmap             : parameter map used for regularisation, default = [];
        %   .voxelSize          : voxel size in mm
        %   .lossFunction       : loss for data fidelity term, 'L1'|'L2'|'MSE', default = 'L1'
        %   .display            : online display the fitting process on figure, true|false, defualt = false
        %   .isPrior            : Estimation of the starting points, default = true
        % 
            fitting2                        = fitting; % copy existing parameter to final output
            fitting2.defaultRegularisation  = true;

            % =====================================================================
            % Optimiser
            % =====================================================================
            if ~isfield(fitting,'optimiser');            fitting2.optimiser            = 'adam';    end
            if ~isfield(fitting,'initialLearnRate');     fitting2.initialLearnRate     = 0.001;     end
            if ~isfield(fitting,'decayRate');            fitting2.decayRate            = 0;         end
            if ~isfield(fitting,'enableComplex');        fitting2.enableComplex        = true;      end
            if ~isfield(fitting,'randomness');           fitting2.randomness           = 0;         end % starting point
            % Parameter space transform
            % 'linear'  : existing rescale01/unscale01 mapping to [0,1] with hard clamping (default, fully backward compatible)
            % 'sigmoid' : logit initialisation + sigmoid recovery; Adam optimises unconstrained z in (-inf,+inf),
            %             eliminating artificial boundary sticking at the cost of modified loss surface geometry near bounds
            if ~isfield(fitting,'parameterTransform'); fitting2.parameterTransform = 'sigmoid'; end

            switch fitting2.optimiser
                case 'adam'
                    if ~isfield(fitting,'adamupdateGradDecay');         fitting2.adamupdateGradDecay        = 0.9;      end
                    if ~isfield(fitting,'adamupdateSqGradDecay');       fitting2.adamupdateSqGradDecay      = 0.999;    end
                    if ~isfield(fitting,'adamupdateEpsilon');           fitting2.adamupdateEpsilon          = 1e-8;     end
                case 'sgdm'
                    if ~isfield(fitting,'sgdmupdateMomentum');          fitting2.sgdmupdateMomentum         = 0.9;      end
                case 'rmsprop'
                    if ~isfield(fitting,'rmspropupdateSqGradDecay');    fitting2.rmspropupdateSqGradDecay   = 0.9;      end
                    if ~isfield(fitting,'rmspropupdateEpsilon');        fitting2.rmspropupdateEpsilon       = 1e-8;     end
            end

            % =====================================================================
            % Loss function
            % =====================================================================
            if ~isfield(fitting,'lossFunction'); fitting2.lossFunction = 'L1'; end  % 'L1'|'L2'|'huber'|'mse'
            if ~isfield(fitting,'tol');          fitting2.tol          = 1e-3; end  % stop if loss < tol

            % =====================================================================
            % Basic stopping criteria (iteration and loss convergence)
            % =====================================================================
            if ~isfield(fitting,'iteration');           fitting2.iteration              = 1e4;                  end    % max. iteration
            if ~isfield(fitting,'convergenceValue');    fitting2.convergenceValue       = 1e-6;                 end    % convergence tolerance
            if ~isfield(fitting,'patience');            fitting2.patience               = 5;                    end    % shared default for all patience counters
            if ~isfield(fitting,'patienceConvergence'); fitting2.patienceConvergence    = fitting2.patience;    end

            % legacy field support
            if isfield(fitting,'Nepoch');               fitting2.iteration = fitting.Nepoch; fitting2 = rmfield(fitting2,'Nepoch'); end

            % =====================================================================
            % Convergence model (v1.1)
            % Controls how the convergence signal is computed from the loss
            % 'linear' : slope of loss over last convergenceWindow iterations (default)
            % 'ema'    : relative change in exponential moving average of loss
            % =====================================================================
            if ~isfield(fitting,'convergenceModel');  fitting2.convergenceModel  = 'ema'; end
            if ~isfield(fitting,'convergenceWindow'); fitting2.convergenceWindow = 20;       end  % used by 'linear' model
            if ~isfield(fitting,'emaDecay');          fitting2.emaDecay          = 0.95;     end  % used by 'ema' model

            % =====================================================================
            % Robust convergence / outlier handling (v1.1)
            % Detects voxels that are not improving relative to the population and
            % downweights their gradient contribution. Convergence signal is computed
            % on the main (non-outlier) population only.
            % =====================================================================
            if ~isfield(fitting,'robustConvergence');      fitting2.robustConvergence      = false;        end
            if ~isfield(fitting,'outlierThresholdMethod'); fitting2.outlierThresholdMethod = 'behaviour';  end  % placeholder for future options
            if ~isfield(fitting,'outlierWeight');          fitting2.outlierWeight          = 0.1;          end  % gradient contribution of outlier voxels
            if ~isfield(fitting,'weightUpdateInterval');   fitting2.weightUpdateInterval   = 5;            end  % iterations between outlier mask updates
            if ~isfield(fitting,'outlierCheckWindow');     fitting2.outlierCheckWindow     = 5;            end  % number of checks for criterion A
            if ~isfield(fitting,'outlierMinFlagDuration'); fitting2.outlierMinFlagDuration = 5;            end  % minimum checks before reinstatement
            if ~isfield(fitting,'outlierVoxelThres');      fitting2.outlierVoxelThres      = 0.01;         end  % criterion A: voxel improvement threshold per check window (1%)
            if ~isfield(fitting,'outlierPopThres');        fitting2.outlierPopThres        = 0.05;         end  % criterion A: median population improvement threshold (5%)
            if ~isfield(fitting,'outlierInitThres');       fitting2.outlierInitThres       = 0.05;         end  % criterion B: voxel improvement threshold from initialisation (5%)
            if ~isfield(fitting,'outlierInitPopThres');    fitting2.outlierInitPopThres    = 0.20;         end  % criterion B: median population improvement threshold from initialisation (20%)

            % =====================================================================
            % Additional convergence signals (v1.1)
            % Independent of robustConvergence. Disabled by default (value = 0).
            % =====================================================================
            if ~isfield(fitting,'convergenceStepTol'); fitting2.convergenceStepTol = 1e-6;                 end  % relative parameter step norm; 0 = disabled
            if ~isfield(fitting,'convergenceGradTol'); fitting2.convergenceGradTol = 1e-6;                 end  % gradient norm; 0 = disabled
            if ~isfield(fitting,'patienceStep');       fitting2.patienceStep       = fitting2.patience; end
            if ~isfield(fitting,'patienceGrad');       fitting2.patienceGrad       = fitting2.patience; end

            % =====================================================================
            % Regularisation
            % =====================================================================
            if ~isfield(fitting,'lambda');    fitting2.lambda    = {0};                 end  % regularisation weight; 0 = no regularisation
            if ~isfield(fitting,'TVmode');    fitting2.TVmode    = '2D';                end  % '2D'|'3D'
            if ~isfield(fitting,'regmap');    fitting2.regmap    = [];                  end  % parameter map(s) to regularise
            if ~isfield(fitting,'voxelSize'); fitting2.voxelSize = [2,2,2];             end  % voxel size in mm
            
            if ~iscell(fitting2.lambda); fitting2.lambda = num2cell(fitting2.lambda);   end

            % =====================================================================
            % Memory management
            % =====================================================================
            if ~isfield(fitting,'isOptimiseMemory'); fitting2.isOptimiseMemory = true; end
            if ~isfield(fitting,'autoMemManage');    fitting2.autoMemManage    = true; end
            if ~isfield(fitting,'segmentOverlap');   fitting2.segmentOverlap   = 0;     end
            if ~isfield(fitting,'NSegmentUser');     fitting2.NSegmentUser     = [];     end


            % =====================================================================
            % Miscellaneous
            % =====================================================================
            if ~isfield(fitting,'outputFilename'); fitting2.outputFilename = [];        end
            if ~isfield(fitting,'ub');             fitting2.ub             = [];        end
            if ~isfield(fitting,'lb');             fitting2.lb             = [];        end
            if ~isfield(fitting,'debug');          fitting2.debug          = false;     end
            if ~isfield(fitting,'isDisplay');      fitting2.isDisplay      = 0;     end
            if ~isfield(fitting,'isMaskedOut');    fitting2.isMaskedOut    = true; end
            
            % =====================================================================
            % Deprecated (kept for backward compatibility, will be removed in v1.2)
            % =====================================================================
            if ~isfield(fitting,'isSampleConsistency'); fitting2.isSampleConsistency = false; end
            if ~isfield(fitting,'isClipGradient');      fitting2.isClipGradient      = 0;     end
            if ~isfield(fitting,'maxGradientThres');    fitting2.maxGradientThres    = 1;     end

            if fitting2.isSampleConsistency
                warning('askadam:deprecated', 'isSampleConsistency is deprecated.');
            end
            if fitting2.isClipGradient
                warning('askadam:deprecated', 'isClipGradient is deprecated.');
            end
            
        end

        function display_basic_fitting_parameters(fitting)
            % Display all active fitting algorithm parameters at the start of optimisation.
            % Sections mirror check_set_default_basic for consistency.
            
            disp('============================');
            disp('AskAdam algorithm parameters');
            disp('============================');
            
            % --- Optimiser ---
            disp('Optimiser');
            disp('---------');
            disp(['Optimiser                    = ' fitting.optimiser]);
            disp(['Initial learning rate        = ' num2str(fitting.initialLearnRate)]);
            disp(['Learning rate decay rate     = ' num2str(fitting.decayRate)]);
            disp(['Allow complex-valued         = ' utils.logical2string(fitting.enableComplex)]);
            disp(['Random initialisation        = ' num2str(fitting.randomness)]);
            disp(['Parameter transform          = ' fitting.parameterTransform]);
            
            switch lower(fitting.optimiser)
                case 'adam'
                    disp(['Adam grad decay              = ' num2str(fitting.adamupdateGradDecay)]);
                    disp(['Adam sq grad decay           = ' num2str(fitting.adamupdateSqGradDecay)]);
                    disp(['Adam epsilon                 = ' num2str(fitting.adamupdateEpsilon)]);
                case 'sgdm'
                    disp(['SGDM momentum                = ' num2str(fitting.sgdmupdateMomentum)]);
                case 'rmsprop'
                    disp(['RMSProp sq grad decay        = ' num2str(fitting.rmspropupdateSqGradDecay)]);
                    disp(['RMSProp epsilon              = ' num2str(fitting.rmspropupdateEpsilon)]);
            end
            
            % --- Loss function ---
            disp(' ');
            disp('Loss function');
            disp('------------');
            disp(['Loss function                = ' fitting.lossFunction]);
            disp(['Loss tolerance               = ' num2str(fitting.tol)]);
            
            % --- Basic stopping criteria ---
            disp(' ');
            disp('Basic stopping criteria');
            disp('-----------------------');
            disp(['Max. iterations              = ' num2str(fitting.iteration)]);
            disp(['Convergence tolerance        = ' num2str(fitting.convergenceValue)]);
            disp(['Patience (convergence)       = ' num2str(fitting.patienceConvergence)]);
            
            % --- Convergence model ---
            disp(' ');
            disp('Convergence model');
            disp('-----------------');
            disp(['Convergence model            = ' fitting.convergenceModel]);
            switch fitting.convergenceModel
                case 'linear'
                    disp(['Convergence buffer size      = ' num2str(fitting.convergenceWindow)]);
                case 'ema'
                    disp(['EMA decay                    = ' num2str(fitting.emaDecay)]);
            end
            
            % --- Robust convergence ---
            disp(' ');
            disp('Robust convergence');
            disp('------------------');
            disp(['Robust convergence           = ' utils.logical2string(fitting.robustConvergence)]);
            if fitting.robustConvergence
                disp(['  Outlier threshold method   = ' fitting.outlierThresholdMethod]);
                disp(['  Outlier weight             = ' num2str(fitting.outlierWeight)]);
                disp(['  Weight update interval     = ' num2str(fitting.weightUpdateInterval)]);
                disp(['  Outlier check window       = ' num2str(fitting.outlierCheckWindow)]);
                disp(['  Min flag duration          = ' num2str(fitting.outlierMinFlagDuration)]);
                disp(['  Criterion A voxel thres    = ' num2str(fitting.outlierVoxelThres)]);
                disp(['  Criterion A pop thres      = ' num2str(fitting.outlierPopThres)]);
                disp(['  Criterion B voxel thres    = ' num2str(fitting.outlierInitThres)]);
                disp(['  Criterion B pop thres      = ' num2str(fitting.outlierInitPopThres)]);
            end
            
            % --- Additional convergence signals ---
            if fitting.convergenceStepTol > 0 || fitting.convergenceGradTol > 0
                disp(' ');
                disp('Additional convergence signals');
                disp('------------------------------');
                if fitting.convergenceStepTol > 0
                    disp(['Step norm tolerance          = ' num2str(fitting.convergenceStepTol)]);
                    disp(['Patience (step)              = ' num2str(fitting.patienceStep)]);
                end
                if fitting.convergenceGradTol > 0
                    disp(['Gradient norm tolerance      = ' num2str(fitting.convergenceGradTol)]);
                    disp(['Patience (gradient)          = ' num2str(fitting.patienceGrad)]);
                end
            end
            
            % --- Regularisation ---
            if fitting.lambda{1} > 0
                disp(' ');
                disp('Regularisation');
                disp('--------------');
                disp(['Regularisation parameter(s)  = ' cell2num2str(fitting.lambda)]);
                if fitting.defaultRegularisation
                    disp(['Regularisation map(s)        = ' cell2str(fitting.regmap)]);
                    disp(['Total variation mode         = ' fitting.TVmode]);
                    disp(['Voxel size (mm)              = ' num2str(fitting.voxelSize)]);
                end
            end
            
            % --- Memory ---
            disp(' ');
            disp('Memory');
            disp('------');
            disp(['Optimise memory              = ' utils.logical2string(fitting.isOptimiseMemory)]);
            disp(['Auto memory management       = ' utils.logical2string(fitting.autoMemManage)]);
            
            disp('============================');
            end
        
        % save the askadam output structure variable into disk space 
        function save_askadam_output(output_filename,out)
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

        %% cost tools

        % compute the cost of Total variation regularisation
        function cost = reg_TV(img,mask,TVmode,voxelSize)
            % voxel_size = [1 1 1];
            % Vr      = 1./sqrt(abs(mask.*askadam.gradient_operator(img,voxel_size)).^2+eps);
            cost = sum(abs(mask.*askadam.gradient_operator(img,voxelSize,TVmode)),4);
            % cost = sqrt(sum(abs(mask.*askadam.gradient_operator(img,voxelSize,TVmode).^2),4));

            % cost    = this.divergence_operator(mask.*(Vr.*(mask.*askadam.gradient_operator(img,voxel_size))),voxel_size);
        end

        % TV regularisation
        function G = gradient_operator(img,voxel_size,TVmode)
            Dx = circshift(img,-1,1) - img;
            Dy = circshift(img,-1,2) - img;
            switch TVmode
                case '2D'
                    G = cat(4,Dx/voxel_size(1),Dy/voxel_size(2));
                case '3D'
                    Dz = circshift(img,-1,3) - img;
                    G = cat(4,Dx/voxel_size(1),Dy/voxel_size(2),Dz/voxel_size(3));
            end
            
        end

        function div = divergence_operator(G,voxel_size)

            G_x = G(:,:,:,1);
            G_y = G(:,:,:,2);
            G_z = G(:,:,:,3);
            
            [Mx, My, Mz] = size(G_x);
            
            Dx = [G_x(1:end-1,:,:); zeros(1,My,Mz)]...
                - [zeros(1,My,Mz); G_x(1:end-1,:,:)];
            
            Dy = [G_y(:,1:end-1,:), zeros(Mx,1,Mz)]...
                - [zeros(Mx,1,Mz), G_y(:,1:end-1,:)];
            
            Dz = cat(3, G_z(:,:,1:end-1), zeros(Mx,My,1))...
                - cat(3, zeros(Mx,My,1), G_z(:,:,1:end-1));
            
            div = -( Dx/voxel_size(1) + Dy/voxel_size(2) + Dz/voxel_size(3) );

        end

        %% Scaling tools

        % undo rescale the network parameters between the defined lower/upper bounds
        % parameterTransform: 'linear' (default) uses unscale01; 'sigmoid' applies sigmoid recovery
        % The sigmoid branch is differentiable for all finite z, preventing boundary sticking.
        function parameters = unscale_parameters(parameters, lb, ub, modelParams, parameterTransform)
            if nargin < 5; parameterTransform = 'linear'; end
            for k = 1:numel(ub)
                if strcmp(parameterTransform, 'sigmoid')
                    % z is unconstrained; sigmoid maps it to (0,1), then scale to (lb, ub)
                    parameters.(modelParams{k}) = lb(k) + (ub(k) - lb(k)) .* (1 ./ (1 + exp(-parameters.(modelParams{k}))));
                else
                    parameters.(modelParams{k}) = askadam.unscale01(parameters.(modelParams{k}), lb(k), ub(k));
                end
            end
        end

        % rescale the network parameters between the defined lower/upper bounds
        % parameterTransform: 'linear' (default) uses rescale01; 'sigmoid' applies logit
        function parameters = rescale_parameters(parameters, lb, ub, modelParams, parameterTransform)
            if nargin < 5; parameterTransform = 'linear'; end
            for k = 1:numel(ub)
                if strcmp(parameterTransform, 'sigmoid')
                    eps_bound  = 1e-4 * (ub(k) - lb(k));
                    theta_clamped = max(min(parameters.(modelParams{k}), ub(k) - eps_bound), lb(k) + eps_bound);
                    tmp_norm   = askadam.rescale01(theta_clamped, lb(k), ub(k));
                    parameters.(modelParams{k}) = log(tmp_norm ./ (1 - tmp_norm));  % logit
                else
                    parameters.(modelParams{k}) = askadam.rescale01(parameters.(modelParams{k}), lb(k), ub(k));
                end
            end
        end

        % rescale input between 0 and 1 given lower and upper bounds
        function img_norm = rescale01(img, lb, ub)
            img_norm = (img - lb) /(ub - lb);
        end
        
        % undo rescale input between 0 and 1 given lower and upper bounds (undo rescale01)
        function img = unscale01(img_norm, lb, ub)
            img = (img_norm * (ub - lb)) + lb;
        end

        % make sure all network parameters stay between 0 and 1
        function parameters = set_boundary01(parameters,enableComplex)

            % TODO: separate real and complex value
            field = fieldnames(parameters);
            for k = 1:numel(field)
                if enableComplex
                    parameters.(field{k})   = max(parameters.(field{k}),0); % Lower bound     
                    parameters.(field{k})   = min(parameters.(field{k}),1); % upper bound
                else
                    parameters.(field{k})   = max(real(parameters.(field{k})),0); % Lower bound     
                    parameters.(field{k})   = min(real(parameters.(field{k})),1); % upper bound
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

        %% optmisation tools

        % (deprecated) prevent to big of the step size due to exploding gradient
        function gradients = clip_gradients(gradients, mask, threshold)
            % get field name
            fields = fieldnames(gradients);
            % loop all parameters
            for k = 1:numel(fields)
                % masking
                gradNorm        = sqrt(sum( utils.vectorise_NDto2D(gradients.(fields{k}),mask) .^2));
                if gradNorm > threshold
                    gradients.(fields{k})(gradients.(fields{k})>threshold) = gradients.(fields{k})(gradients.(fields{k})>threshold) * (threshold / gradNorm);
                    % gradients.(fields{k}) = gradients.(fields{k}) * (threshold / gradNorm);
                end
            end

        end

        % (deprecated) prevent to big of the step size due to exploding gradient
        function [gradients, movingAvgNorm] = adaptive_gradient_clipping(gradients, mask, movingAvgNorm, movingAvgFactor, maxGradientThres)

            if nargin < 5
                maxGradientThres = 1;
            end

            % get field name
            fields = fieldnames(gradients);

            if isempty(movingAvgNorm); for k = 1:numel(fields); movingAvgNorm.(fields{k}) = maxGradientThres; end; end


            % loop all parameters
            for k = 1:numel(fields)
                % masking and get gradient norm
                gradNorm        = sqrt(sum( gradients.(fields{k}) .^2));
                % compute moving norm
                movingAvgNorm.(fields{k})   = movingAvgFactor .* movingAvgNorm.(fields{k}) + (1 - movingAvgFactor) .* gradNorm;
                gradientThreshold           = min(movingAvgNorm.(fields{k}),maxGradientThres); % movingAvgNorm.(fields{k}); Update threshold

                if any(gradNorm > gradientThreshold)
                    mask_outliers   = gradients.(fields{k}) > gradientThreshold;
                    tmp             = gradients.(fields{k}) .* gradientThreshold ./ gradNorm;
                    gradients.(fields{k})(mask_outliers) = tmp(mask_outliers);
                end
            end

        end

        % learning rate update formulism
        function learnRate = update_learn_rate(initialLearnRate,decayRate, epoch)
            learnRate = initialLearnRate ./ (1 + decayRate*(epoch-1));
        end

        % compute convergence value based on linear fit
        function [convergenceCurr, convergenceBuffer] = update_convergence(convergenceBuffer)
            A = [(1:numel(convergenceBuffer)).', ones(numel(convergenceBuffer),1)]; % A matrix to derive convergence

            mc                  = A\convergenceBuffer;  % linear fit y = mx + c
            convergenceCurr     = -mc(1);               % slope
        end

        function [ema_curr, convergenceCurr] = update_convergence_ema(loss_curr, ema_prev, decay)
        % Purpose: Replace the linear slope convergence signal with an exponential moving average (EMA) of the loss. 
        % The EMA smooths out short-term oscillations that can cause premature stopping under the current linear slope approach. 
        % The relative change in EMA (rather than absolute slope) is used as the convergence signal, making it scale-invariant 
        % across different loss magnitudes, models, and regularisation weights.
        % loss_curr : scalar loss value at current iteration
        % ema_prev  : EMA value from previous iteration
        % decay     : EMA decay factor (fitting.emaDecay)
        % ema_curr  : updated EMA value, stored for next iteration
        % convergenceCurr : relative change in EMA, used as convergence signal
            ema_curr = decay * ema_prev + (1 - decay) * loss_curr;
            convergenceCurr = abs(ema_curr - ema_prev) / (abs(ema_prev) + 1e-8);
        end

        %% DEBUG tools

        function isNaNInf = check_nan_in_gradients(gradients, mask)

            isNaNInf = false;

            % get field name
            fields = fieldnames(gradients);
            % loop all parameters
            for k = 1:numel(fields)
                % masking
                gradNorm = sqrt(sum( utils.vectorise_NDto2D(gradients.(fields{k}),mask) .^2));

                isNaNInf = or(or(isNaNInf,isnan(gradNorm)),isinf(gradNorm));
            end

             if isNaNInf; disp('Gradients have NaN(s)!'); end
        end

        function lineLoss = setup_display
            figure
            C = colororder;
            lineLoss = animatedline('Color',C(2,:));
            ylim([0 inf])
            xlabel("Iteration")
            ylabel("Loss")
            grid on
        end

        function add_point_to_display(lineLoss,epoch,loss,start)

            addpoints(lineLoss,epoch, loss);
                    
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
            drawnow
        end

    end

end