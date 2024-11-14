classdef askadam < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% 
% This is the class of all askadam realted functions
%
% Date created: 4 April 2024 
% Date modified: 21 August 2024
% Date modified: 3 October 2024

    properties (Constant)
        epsilon = 1e-8;
    end
%
    properties (GetAccess = public, SetAccess = protected)

    end

    methods

        function [gradients,loss,loss_fidelity,loss_reg,residuals] = model_gradient(this, parameters, data, mask, weights, fitting, FWDfunc, varargin)
        % Input
        % ----------
        % parameters    : Structure variable containing all parameters to be estimated
        % data          : N-D measureemnt data
        % mask          : M-D signal mask (M=[1,3])
        % weights       : N-D weights for optimisaiton
        % fitting       : Structure variable containing all fitting algorithm setting
        % FWDfunc       : function handle of forward model
        % varargin      : contains additional input requires for FWDfunc
        % 
        % Output
        % ------
        % gradients     : Adam gradient
        % loss          : total loss
        % loss_fidelity : loss associated with data fidelity (consistancy)
        % loss_reg      : loss associated with (TV) regularisation

            % Forward signal simulation
            signal_FWD = FWDfunc(this.unscale_parameters(parameters,fitting.lb,fitting.ub,fitting.modelParams),varargin{:});
            % masking Forward signal if the 'signal_FWD' is not 2D
            if ~ismatrix(signal_FWD); signal_FWD = utils.reshape_ND2AD(signal_FWD, mask); end
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
            loss_reg = 0;
            if fitting.lambda{1} > 0
                for kreg = 1:numel(fitting.lambda)
                    Nsample     = numel(mask(mask ~= 0));

                    cost        = this.reg_TV(utils.reshape_AD2ND(parameters.(fitting.regmap{kreg}),mask),mask,fitting.TVmode,fitting.voxelSize);
                    loss_reg    = sum(abs(cost),"all")/Nsample *fitting.lambda{kreg} + loss_reg;
                end
            end
            
            % compute loss
            loss = loss_fidelity + loss_reg;
            
            % Calculate gradients with respect to the learnable parameters.
            gradients = dlgradient(loss,parameters);

        end

        % askAdam optimisation loop
        function out = optimisation(this, data, mask, weights, parameters, fitting, FWDfunc, varargin)
        % Input
        % -----
        % data                  : 2-D (vectorised imaging) data
        % mask                  : (1-3)D signal mask applied on FWDfunc, NOTE this mask does NOT apply on data
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
        % FWDfunc               : function handle for forward signal generation
        % varargin              : additional input for FWDfunc other than 'parameter' and 'mask' (same order as FWDfunc)
        %
        % Output
        % ------
        % out                   : structure contains optimisation result
        %

            dims = size(mask,1:3); mask_idx = find(mask>0);

            %%%%%%%%%%%%%%%%%%%%%%%%%% 1. I/O Setup %%%%%%%%%%%%%%%%%%%%%%%%%%
            % data can be either 2D or ND, if ND then convert to 2D here
            % masking if the input data are not 2D
            if ~ismatrix(data);     data    = utils.reshape_ND2AD(data,      mask_idx); else; data = data(:,mask_idx);     end
            if ~ismatrix(weights);  weights = utils.reshape_ND2AD(weights,   mask_idx); elseif ~isempty(weights); weights = weights(:,mask_idx);  end
                
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
            % check and set fitting default
            fitting = this.check_set_default_basic(fitting);
            
            % initiate starting points arrays
            parameters = this.set_boundary01(this.initialise_parameter(dims,parameters,fitting,mask));    % the parameter maps here are normalised to [0,1] using their boundary values

            % clear cache before running everthing
            accfun = dlaccelerate(@this.model_gradient); clearCache(accfun)

            % initiate optimiser parameters
            averageGrad = []; averageSqGrad = []; vel = [];

            % parameter for gradient check
            movingAvgFactor             = 0.9; % Moving average factor
            movingAvgNorm               = []; %0; % Initialize moving average of gradient norms
            epochsWithoutImprovement    = 0;
            
            % create buffer arrays
            convergenceBuffer       = ones(fitting.convergenceWindow,1);
            insepctInterval         = 5;  % interval to check loss on each voxel
            NparamBuffer            = 5; kBuffer = 1;
            parameterBuffer         = repmat({parameters},1,NparamBuffer);

            if fitting.isDisplay; lineLoss = this.setup_display; end

            % compute the loss and residual at the starting point
            [~,loss,loss_fidelity,loss_reg,residuals] = dlfeval(accfun,parameters,data,mask,weights,fitting,FWDfunc,varargin{:});
            lossAll0                = extractdata(mean(reshape(residuals,Nmeas,Nvol),1));
            loss                    = double(utils.dlarray2single(loss));
            minLoss                 = loss;
            minLossFidelity         = utils.dlarray2single(loss_fidelity);
            minLossRegularisation   = utils.dlarray2single(loss_reg);
            minResiduals            = residuals;
            parameters_minLoss      = parameters;
            minIteration            = 0;
            epoch                   = 0;

            start = tic;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%% 3. Optimisation %%%%%%%%%%%%%%%%%%%%%%%%%%
            if fitting.iteration > 0
                
                % display optimisation algorithm parameters
                this.display_basic_fitting_parameters(fitting);

                disp('Optimisation begins...');
                disp('----------------------');

                for epoch = 1:fitting.iteration
                    
                    %%%%%%%%%%%%%%%%%%%% 3.1. Model evaluation module %%%%%%%%%%%%%%%%%%%%
                    parameters = this.set_boundary01(parameters); % make sure the parameters are [0,1]
                    
                    [gradients,loss,loss_fidelity,loss_reg,residuals] = dlfeval(accfun,parameters,data,mask,weights,fitting,FWDfunc,varargin{:}); % Evaluate the model gradients and loss using dlfeval and the modelGradients function
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    if fitting.debug; isNaNInf = this.check_nan_in_gradients(gradients, mask_idx); if isNaNInf; disp(num2str(epoch));end; end % DEBUG module
                    
                    %%%%%%%%%%%%%%%%%%%% 3.2. Stopping criteria module %%%%%%%%%%%%%%%%%%%%
                    loss = double(utils.dlarray2single(loss)); % get loss

                    % store the results with minimal loss
                    if minLoss > loss
                        minLoss                 = loss;
                        minLossFidelity         = loss_fidelity;
                        minLossRegularisation   = loss_reg;
                        minResiduals            = residuals;
                        parameters_minLoss      = parameters;
                        minIteration            = epoch;
                        
                    end

                    % Update convergence value
                    [convergenceCurr, convergenceBuffer] = this.update_convergence([convergenceBuffer(2:end);loss]); 

                    % check if there is any global improvement
                    if convergenceCurr > fitting.convergenceValue || epoch <= fitting.convergenceWindow
                        
                        epochsWithoutImprovement = 0; % when global loss gradient > tolerance, -> improving, then reset epochsWithoutImprovement

                    elseif (minLoss - loss) > fitting.convergenceValue 
                        
                        epochsWithoutImprovement = 0; % if the current loss is smaller than the minimum loss by the convergenceValue then suppose it's improving so can reset
                    else
                        epochsWithoutImprovement = epochsWithoutImprovement + 1;
                    end
    
                    % check if the optimisation should be stopped
                    if epochsWithoutImprovement > fitting.patience
                        fprintf('Optimisation is done. No significant improvements as defined in fitting setup. \n');
                        break
                    end
                    if loss < fitting.tol
                        fprintf('Optimisation is done. Loss is less than the tolerance %e \n',fitting.tol);
                        break
                    end
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    %%%%%%%%%%%%%%%%%%%% 3.3 Individual sample consistency module %%%%%%%%%%%%%%%%%%%%
                    if fitting.isClipGradient

                        % clipping gradients to avoid sudden big jumps
                        [gradients, movingAvgNorm] = this.adaptive_gradient_clipping(gradients, mask_idx, movingAvgNorm, movingAvgFactor, fitting.maxGradientThres); % gradientThreshold = 1; gradients = this.clip_gradients(gradients, mask, gradientThreshold);
    
                    end

                    if fitting.isSampleConsistency
                        
                        % preparing buffer
                        if kBuffer < NparamBuffer
                            % compute loss on each voxel and compare to previous loss
                            lossAll         = extractdata(mean(reshape(residuals,Nmeas,Nvol),1));
                            maskNoImprove   = lossAll > lossAll0;
                            for kf = 1:numel(fitting.modelParams)
                                % replace the no-improvement voxel with previous position
                                parameters.(fitting.modelParams{kf})(maskNoImprove) = parameterBuffer{kBuffer}.(fitting.modelParams{kf})(maskNoImprove);
                            end
                            % update loss
                            lossAll(maskNoImprove)  = lossAll0(maskNoImprove);
                            lossAll0                = lossAll;   
                            % update buffer
                            parameterBuffer(1:end-1) = parameterBuffer(2:end); parameterBuffer(end) = {parameters};
                            kBuffer = kBuffer + 1;
                        end


                        % check if the loss of the each voxel gets improved every 5 iterations
                        if mod(epoch,insepctInterval) == 0 

                            % once we have sufficient short term memory, check if the fitting actually improving the loss
                            if epoch > insepctInterval*NparamBuffer && epoch < fitting.iteration

                                % compute loss on each voxel and compare to previous loss
                                lossAll         = extractdata(mean(reshape(residuals,Nmeas,Nvol),1));
                                maskNoImprove   = lossAll > lossAll0;
    
                                for kf = 1:numel(fitting.modelParams)
                                    % draw a random number
                                    n = randi(NparamBuffer);  
                                    % replace the no-improvement voxel with one of those in the buffer for restart
                                    parameters.(fitting.modelParams{kf})(maskNoImprove) = parameterBuffer{n}.(fitting.modelParams{kf})(maskNoImprove);
                                end
                                % update loss
                                lossAll0 = lossAll;          

                                 % update buffer
                                parameterBuffer(1:end-1) = parameterBuffer(2:end); parameterBuffer(end) = {parameters};
                            end
                        end
                    end
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
                    %%%%%%%%%%%%%%%%%%%% 3.4. Parameter update module %%%%%%%%%%%%%%%%%%%%
                    % Update learning rate
                    learningRate = this.update_learn_rate(fitting.initialLearnRate,fitting.decayRate, epoch); 
                    % Update the network parameters using the one of the following optimisers
                    if epoch < fitting.iteration
                        switch lower(fitting.optimiser)
                            case 'adam'
                                [parameters,averageGrad,averageSqGrad]  = adamupdate(parameters,gradients,averageGrad, ...
                                                                                        averageSqGrad,epoch,learningRate,fitting.adamupdateGradDecay,fitting.adamupdateSqGradDecay,fitting.adamupdateEpsilon);
                            case 'sgdm'
                                [parameters,vel]                        = sgdmupdate(parameters,gradients,vel, ...
                                                                                        learningRate,fitting.sgdmupdateMomentum);
                            case 'rmsprop'
                                [parameters,averageSqGrad]              = rmspropupdate(parameters,gradients,averageSqGrad, ...
                                                                                        learningRate,fitting.rmspropupdateSqGradDecay,fitting.rmspropupdateEpsilon);
                        end
                    end
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    if fitting.isDisplay; this.add_point_to_display(lineLoss,epoch,loss,start); end % DEBUG module

                    %%%%%%%%%%%%%%%%%%%% verbose module %%%%%%%%%%%%%%%%%%%%
                    if mod(epoch,100) == 0 || epoch == 1
                        
                        D = duration(0,0,toc(start),'Format','hh:mm:ss');
                        if fitting.debug
                            loss_nooutliers = extractdata(mean(reshape(residuals,Nmeas,Nvol),1));
                            mask_outliers   = isoutlier(loss_nooutliers);
                            loss_nooutliers = gather( mean(loss_nooutliers(mask_outliers==0)));

                            fprintf('Iteration #%4d,     Loss = %f,     Loss (w/o outliers) = %f,   Convergence = %e,     Elapsed:%s \n',epoch,loss,loss_nooutliers,convergenceCurr,string(D));
                        else
                            fprintf('Iteration #%4d,     Loss = %.3e,     Convergence = %.3e,       Learn rate = %.3e,     Elapsed:%s \n',epoch,loss,convergenceCurr, learningRate, string(D));
                        end
                    end
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%%%%%%%% 4. Finalisation %%%%%%%%%%%%%%%%%%%%%%%%%%
            % display final message
            if fitting.iteration > 0
                fprintf('Final loss         =  %e\n',double(loss));
                fprintf('Final convergence  =  %e\n',double(convergenceCurr));
                fprintf('Final #iterations  =  %d\n',epoch);
            end
            % make sure the final results stay within boundary
            parameters         = this.set_boundary01(parameters);
            parameters_minLoss = this.set_boundary01(parameters_minLoss);

            parameters          = utils.undo_masking_ND2AD_preserve_struct(parameters,mask);
            parameters_minLoss  = utils.undo_masking_ND2AD_preserve_struct(parameters_minLoss,mask);
            
            % rescale the network parameters
            parameters          = this.unscale_parameters(parameters,           fitting.lb,fitting.ub,fitting.modelParams);
            parameters_minLoss  = this.unscale_parameters(parameters_minLoss,   fitting.lb,fitting.ub,fitting.modelParams);
            for k = 1:numel(fitting.modelParams)
                % final iteration result
                tmp = utils.dlarray2single(parameters.(fitting.modelParams{k}) .* mask); 
                out.final.(fitting.modelParams{k}) = tmp;

                % minimum loss result
                tmp = utils.dlarray2single(parameters_minLoss.(fitting.modelParams{k}) .* mask); 
                out.min.(fitting.modelParams{k}) = tmp;
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
                    % random initialisation
                    tmp =   rand(size(pars0.(modelParams{k})),'single') ;     % values between [0,1]
                    tmp =  (1-randomness)* this.rescale01(pars0.(modelParams{k}), lb(k), ub(k)) + randomness*tmp;     % values between [0,1]
                else
                     % random initialisation
                    tmp = rand(img_size,'single') ;     % values between [0,1]

                end
                % put it into dlarray
                parameters.(modelParams{k}) = gpuArray( dlarray( tmp ));
            end

            % masking
            parameters = utils.masking_ND2AD_preserve_struct(parameters,mask); %parameters = utils.reshape_ND2AD_struct(parameters,mask);% structfun(@transpose,utils.vectorise_NDto2D_struct(parameters,mask),'UniformOutput',false);
            
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
        %   .regmap             : parameter map used for regularisation, 'fa'|'ra'|'Da'|'De', default = 'fa'
        %   .voxelSize          : voxel size in mm
        %   .lossFunction       : loss for data fidelity term, 'L1'|'L2'|'MSE', default = 'L1'
        %   .display            : online display the fitting process on figure, true|false, defualt = false
        %   .isPrior            : Estimation of the starting points, default = true
        % 
            fitting2 = fitting;

            % get fitting algorithm setting
            if ~isfield(fitting,'iteration');           fitting2.iteration = 4000;          end
            if isfield(fitting,'Nepoch') % legacy
                fitting2.iteration  = fitting.Nepoch; fitting2            = rmfield(fitting2,'Nepoch');
            end
            if ~isfield(fitting,'initialLearnRate');    fitting2.initialLearnRate   = 0.001;    end
            if ~isfield(fitting,'decayRate');           fitting2.decayRate          = 0;        end
            if ~isfield(fitting,'optimiser');           fitting2.optimiser          = 'adam';   end
            if ~isfield(fitting,'tol');                 fitting2.tol                = 1e-3;     end
            if ~isfield(fitting,'lambda');              fitting2.lambda             = {0};      end
            if ~isfield(fitting,'TVmode');              fitting2.TVmode             = '2D';     end
            if ~isfield(fitting,'voxelSize');           fitting2.voxelSize          = [2,2,2];  end
            if ~isfield(fitting,'isDisplay');           fitting2.isDisplay          = 0;        end
            if ~isfield(fitting,'randomness');          fitting2.randomness         = 0;        end
            if ~isfield(fitting,'convergenceValue');    fitting2.convergenceValue   = 1e-8;     end
            if ~isfield(fitting,'convergenceWindow');   fitting2.convergenceWindow  = 20;       end
            if ~isfield(fitting,'lossFunction');        fitting2.lossFunction       = 'L1';     end
            if ~isfield(fitting,'outputFilename');      fitting2.outputFilename     = [];       end
            if ~isfield(fitting,'ub');                  fitting2.ub                 = [];       end
            if ~isfield(fitting,'lb');                  fitting2.lb                 = [];       end
            if ~isfield(fitting,'debug');               fitting2.debug              = false;    end
            if ~isfield(fitting,'patience');            fitting2.patience           = 5;        end
            if ~isfield(fitting,'isSampleConsistency'); fitting2.isSampleConsistency= false;    end
            if ~isfield(fitting,'isClipGradient');      fitting2.isClipGradient     = 0;        end
            if ~isfield(fitting,'maxGradientThres');    fitting2.maxGradientThres   = 1;        end
            
            if ~iscell(fitting2.lambda);                fitting2.lambda = num2cell(fitting2.lambda); end

            switch fitting2.optimiser
                case 'adam'
                    if ~isfield(fitting,'adamupdateGradDecay');         fitting2.adamupdateGradDecay        = .9;    end
                    if ~isfield(fitting,'adamupdateSqGradDecay');       fitting2.adamupdateSqGradDecay      = .999;  end
                    if ~isfield(fitting,'adamupdateEpsilon');           fitting2.adamupdateEpsilon          = 1e-8;  end
                case 'sgdm'
                    if ~isfield(fitting,'sgdmupdateMomentum');          fitting2.sgdmupdateMomentum         = .9;    end
                case 'rmsprop'
                    if ~isfield(fitting,'rmspropupdateSqGradDecay');    fitting2.rmspropupdateSqGradDecay   = .9;    end
                    if ~isfield(fitting,'rmspropupdateEpsilon');        fitting2.rmspropupdateEpsilon       = 1e-8;  end
            end
            
        end

        % display fitting algorithm parameters
        function display_basic_fitting_parameters(fitting)
            % display optimisation algorithm parameters
            disp('============================');
            disp('AskAdam algorithm parameters');
            disp('============================');
            disp('Optimisation setup');
            disp('------------------');
            disp(['Optimiser                = ' num2str(fitting.optimiser)]);
            disp(['Initial learning rate    = ' num2str(fitting.initialLearnRate)]);
            disp(['Learning rate decay rate = ' num2str( fitting.decayRate)]);
            disp(['Max. #iterations         = ' num2str(fitting.iteration)]);
            disp('--------------------------');
            disp('Loss and stopping criteria');
            disp('--------------------------');
            disp(['Loss function            = ' fitting.lossFunction]);
            disp(['Loss tolerance           = ' num2str(fitting.tol)]);
            disp(['Convergence tolerance    = ' num2str(fitting.convergenceValue)]);
            disp(['Convergence buffer size  = ' num2str(fitting.convergenceWindow)]);
            disp(['Patience                 = ' num2str( fitting.patience)]);
            if fitting.lambda{1} > 0 
                disp(['Regularisation parameter(s) = ' cell2num2str(fitting.lambda)]);
                disp(['Regularisation Map(s)       = ' cell2str(fitting.regmap)]);
                disp(['Total variation mode        = ' fitting.TVmode]);
            end
            disp('-----------------------------');
            disp('Individual sample consistency');
            disp('-----------------------------');
            disp(['Check sample consistency = ' utils.logical2string(fitting.isSampleConsistency)]);
            disp(['Clip gradients           = ' utils.logical2string(fitting.isClipGradient)]);
            if fitting.isClipGradient
                disp(['Max. gradient threshold  = ' num2str( fitting.maxGradientThres)]);
            end
            disp('-----------------------------');
            
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

        %% scalling tools

        % undo rescale the network parameters between the defined lower/upper bounds
        function parameters = unscale_parameters(parameters,lb,ub,modelParams)
            for k = 1:numel(ub)
                parameters.(modelParams{k}) = askadam.unscale01(parameters.(modelParams{k}), lb(k), ub(k));
            end

        end

        % rescale the network parameters between the defined lower/upper bounds
        function parameters = rescale_parameters(parameters,lb,ub,modelParams)
            for k = 1:numel(ub)
                parameters.(modelParams{k}) = askadam.rescale01(parameters.(modelParams{k}), lb(k), ub(k));
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
        function parameters = set_boundary01(parameters)

            field = fieldnames(parameters);
            for k = 1:numel(field)
                parameters.(field{k})   = max(parameters.(field{k}),0); % Lower bound     
                parameters.(field{k})   = min(parameters.(field{k}),1); % upper bound

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

        % prevent to big of the step size due to exploding gradient
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

        % prevent to big of the step size due to exploding gradient
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