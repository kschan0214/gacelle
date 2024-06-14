classdef askadam < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% 
% This is the class of all askadam realted functions
%
% Date created: 4 April 2024 
% Date modified:
    properties (GetAccess = public, SetAccess = protected)

    end

    methods

        % askAdam optimisation loop
        function out = optimisation(this, pars0, ModelGradientFunc, data,mask,weights, fitting,varargin)
        % Input
        % -----
        % ModelGradientFunc     : function handle for gradient descent
        % fitting               : structure contains fitting algorithm parameters
        % pars0                 : (optional) initial starting points of model parameters
        % data                  : N-D imaging data
        % mask                  : 2/3D signal mask
        % varargin              : other input for ModelGradientFunc (same order as ModelGradientFunc)
        %
        % Output
        % ------
        % out                   : structure contains optimisation result
        %
            dims = size(data);

            parameters = this.initialise_model(dims(1:3),pars0,fitting.ub,fitting.lb,fitting);

            % clear cache before running everthing
            accfun = dlaccelerate(ModelGradientFunc);
            clearCache(accfun)

            % optimisation process
            averageGrad     = [];
            averageSqGrad   = [];
            
            if fitting.isdisplay
                figure
                C = colororder;
                lineLoss = animatedline('Color',C(2,:));
                ylim([0 inf])
                xlabel("Iteration")
                ylabel("Loss")
                grid on
                
            end
            start = tic;

            minLoss                 = inf; 
            minLossFidelity         = 0; 
            minLossRegularisation   = 0;
            convergenceCurr         = 1+fitting.convergenceValue;
            convergenceBuffer       = ones(fitting.convergenceWindow,1);
            A                       = [(1:fitting.convergenceWindow).', ones(fitting.convergenceWindow,1)]; % A matrix to derive convergence
            % optimisation
            for epoch = 1:fitting.Nepoch
                
                % make sure the parameters are [0,1]
                parameters = this.set_boundary(parameters);

                % Evaluate the model gradients and loss using dlfeval and the modelGradients function.
                [gradients,loss,loss_fidelity,loss_reg] = dlfeval(accfun,parameters,data,mask,weights,fitting,varargin{:});
            
                % Update learning rate.
                learningRate = fitting.initialLearnRate / (1+ fitting.decayRate*epoch);
                
                % get loss and compute convergence value
                loss                = double(gather(extractdata(loss)));
                convergenceBuffer   = [convergenceBuffer(2:end);loss];
                mc                  = A\convergenceBuffer;
                convergenceCurr     = -mc(1);

                % store also the results with minimal loss
                if minLoss > loss
                    minLoss                 = loss;
                    minLossFidelity         = loss_fidelity;
                    minLossRegularisation   = loss_reg;
                    parameters_minLoss      = parameters;
                end
                % check if the optimisation should be stopped
                if convergenceCurr < fitting.convergenceValue && epoch >= fitting.convergenceWindow
                    fprintf('Convergence is less than the tolerance %e \n',fitting.convergenceValue);
                    break
                end
                if loss < fitting.tol
                    fprintf('Loss is less than the tolerance %e \n',fitting.tol);
                    break
                end

                % Update the network parameters using the adamupdate function.
                [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
                    averageSqGrad,epoch,learningRate);
                
                
                if fitting.isdisplay
                    
                    addpoints(lineLoss,epoch, loss);
                
                    D = duration(0,0,toc(start),'Format','hh:mm:ss');
                    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
                    drawnow
                end
                if mod(epoch,100) == 0 || epoch == 1
                    % display some info
                    D = duration(0,0,toc(start),'Format','hh:mm:ss');
                    fprintf('Iteration #%4d,     Loss = %f,      Convergence = %e,     Elapsed:%s \n',epoch,loss,convergenceCurr,string(D));
                end
                
            end
            fprintf('Final loss         =  %e\n',double(loss));
            fprintf('Final convergence  =  %e\n',double(convergenceCurr));
            fprintf('Final #iterations  =  %d\n',epoch);

            % make sure the final results stay within boundary
            parameters = this.set_boundary(parameters);
            
            % rescale the network parameters
            parameters          = this.rescale_parameters(parameters,           fitting.lb,fitting.ub,fitting.model_params);
            parameters_minLoss  = this.rescale_parameters(parameters_minLoss,   fitting.lb,fitting.ub,fitting.model_params);
            for k = 1:numel(fitting.model_params)
                % final iteration result
                tmp = single(gather(extractdata(parameters.(fitting.model_params{k}) .* mask(:,:,:,1)))); 
                out.final.(fitting.model_params{k}) = tmp;

                % minimum loss result
                tmp = single(gather(extractdata(parameters_minLoss.(fitting.model_params{k}) .* mask(:,:,:,1)))); 
                out.min.(fitting.model_params{k}) = tmp;
            end
            out.final.loss          = loss;
            out.final.loss_fidelity = double(gather(extractdata(loss_fidelity)));
            out.min.loss            = minLoss;
            out.min.loss_fidelity   = double(gather(extractdata(minLossFidelity)));
            if fitting.lambda == 0
                out.final.loss_reg  = 0;
                out.min.loss_reg    = 0;
            else
                out.final.loss_reg  = double(gather(extractdata(loss_reg)));
                out.min.loss_reg    = double(gather(extractdata(minLossRegularisation)));
            end

        end

        % initialise network parameters
        function parameters = initialise_model(this,img_size,pars0,ub,lb,fitting)
            
            % get relevant parameters
            randomness      = fitting.randomness;
            model_params    = fitting.model_params;

            for k = 1:numel(model_params)
                % random initialisation
                tmp = rand(img_size,'single') ;     % values between [0,1]
                % if starting points are provided
                if ~isempty(pars0)
                    tmp =  (1-randomness)* this.rescale01(pars0.(model_params{k}), lb(k), ub(k)) + randomness*tmp;     % values between [0,1]
                end
                % put it into dlarray
                parameters.(model_params{k}) = gpuArray( dlarray( tmp ));
            end


        end
   
        % rescale the network parameters between the defined lower/upper bounds
        function parameters = rescale_parameters(this,parameters,lb,ub,model_params)
            for k = 1:numel(ub)
                parameters.(model_params{k}) = this.unscale01(parameters.(model_params{k}), lb(k), ub(k));
            end

        end

        % compute the cost of Total variation regularisation
        function cost = reg_TV(this,img,mask,TVmode,voxelSize)
            % voxel_size = [1 1 1];
            % Vr      = 1./sqrt(abs(mask.*askadam.gradient_operator(img,voxel_size)).^2+eps);
            cost = sum(abs(mask.*this.gradient_operator(img,voxelSize,TVmode)),4);

            % cost    = this.divergence_operator(mask.*(Vr.*(mask.*askadam.gradient_operator(img,voxel_size))),voxel_size);
        end
    

    end

    methods(Static)

        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default_basic(fitting)
        % Input
        % -----
        % fitting               : structure contains fitting algorithm parameters
        %   .Nepoch             : no. of maximum iterations, default = 4000
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
            if ~isfield(fitting,'Nepoch')
                fitting2.Nepoch = 4000;
            end
            if ~isfield(fitting,'initialLearnRate')
                fitting2.initialLearnRate = 0.01;
            end
            if ~isfield(fitting,'decayRate')
                fitting2.decayRate = 0.0005;
            end
            if ~isfield(fitting,'tol')
                fitting2.tol = 1e-3;
            end
            if ~isfield(fitting,'lambda')
                fitting2.lambda = 0;
            end
            if ~isfield(fitting,'TVmode')
                fitting2.TVmode = '2D';
            end
            if ~isfield(fitting,'voxelSize')
                fitting2.voxelSize = [2,2,2];
            end
            if ~isfield(fitting,'isdisplay')
                fitting2.isdisplay = 0;
            end
            if ~isfield(fitting,'randomness')
                fitting2.randomness = 0;
            end
            if ~isfield(fitting,'convergenceValue')
                fitting2.convergenceValue = 1e-8;
            end
            if ~isfield(fitting,'convergenceWindow')
                fitting2.convergenceWindow = 20;
            end
            if ~isfield(fitting,'lossFunction')
                fitting2.lossFunction = 'L1';
            end
            if ~isfield(fitting,'output_filename')
                fitting2.output_filename = [];
            end
            if ~isfield(fitting,'isPrior')
                fitting2.isPrior = true;
            end
            if ~isfield(fitting,'ub')
                fitting2.ub = [];
            end
            if ~isfield(fitting,'lb')
                fitting2.lb = [];
            end

        end

        % make sure all network parameters stay between 0 and 1
        function parameters = set_boundary(parameters)

            field = fieldnames(parameters);
            for k = 1:numel(field)
                parameters.(field{k})   = max(parameters.(field{k}),0); % Lower bound     
                parameters.(field{k})   = min(parameters.(field{k}),1); % upper bound

            end

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

        % display fitting algorithm parameters
        function display_basic_fitting_parameters(fitting)
            % display optimisation algorithm parameters
            disp('----------------------------');
            disp('AskAdam algorithm parameters');
            disp('----------------------------');
            disp(['Maximum no. of iteration = ' num2str(fitting.Nepoch)]);
            disp(['Loss function            = ' fitting.lossFunction]);
            disp(['Loss tolerance           = ' num2str(fitting.tol)]);
            disp(['Convergence tolerance    = ' num2str(fitting.convergenceValue)]);
            disp(['Initial learning rate    = ' num2str(fitting.initialLearnRate)]);
            disp(['Learning rate decay rate = ' num2str( fitting.decayRate)]);
            if fitting.lambda > 0 
                disp(['Regularisation parameter = ' num2str(fitting.lambda)]);
                disp(['Regularisation Map       = ' fitting.regmap]);
                disp(['Total variation mode     = ' fitting.TVmode]);
            end
        end

        % rescale input between 0 and 1 given lower and upper bounds
        function img_norm = rescale01(img, lb, ub)
            img_norm = (img - lb) /(ub - lb);
        end
        % undo rescale input between 0 and 1 given lower and upper bounds
        function img = unscale01(img_norm, lb, ub)
            img = (img_norm * (ub - lb)) + lb;
        end

        % make sure data does not contain any NaN/Inf and update mask
        function [data,mask] = remove_img_naninf(data,mask)
        % Input
        % -------
        % data  : N-D image that may or may not contains NaN or Inf
        % mask  : 2D/3D mask
        %
        % Output
        % -------
        % data  : N-D image that is free from NaN or Inf
        % mask  : 2/3D mask that excludes NaN or Inf voxels
        %
            % mask sure no nan or inf
            Nvoxel_old              = numel(mask(mask>0));
            mask_nonnaninf          = and(~isnan(data) , ~isinf(data));
            data(mask_nonnaninf==0)  = 0;
            data(mask_nonnaninf==0)  = 0;
            for k = 4:ndims(data)
                mask_nonnaninf          = min(mask_nonnaninf,[],k);
            end
            mask                    = and(mask,mask_nonnaninf);
            Nvoxel_new              = numel(mask(mask>0));
            if Nvoxel_old ~= Nvoxel_new
                disp('The mask is updated due to the presence of NaN/Inf. Please make use of the output mask in your subseqeunt analysis.');
            end
        end

        % determine how the dataset will be divided based on vailable memory in GPU
        function [NSegment,maxSlice] = find_optimal_divide(mask,memoryFixPerVoxel,memoryDynamicPerVoxel)
        % Input
        % -----
        % mask                  : 3D signal mask
        % memoryFixPerVoxel     : memory usage 
        %
            % % get these number based on mdl fit
            % memoryFixPerVoxel       = 0.0013;
            % memoryDynamicPerVoxel   = 0.05;

            dims = size(mask);

            % GPU info
            gpu         = gpuDevice;    
            maxMemory   = floor(gpu.TotalMemory / 1024^3)*1024^3 / (1024^2);        % Mb

            % find max. memory required
            memoryRequiredFix       = memoryFixPerVoxel * prod(dims(1:3)) ;         % Mb
            memoryRequiredDynamic   = memoryDynamicPerVoxel * numel(mask(mask>0));  % Mb

            if maxMemory > (memoryRequiredFix + memoryRequiredDynamic)
                % if everything fit in GPU
                maxSlice = dims(3);
                NSegment = 1;
            else
                % if not then divide the data
                 NvolSliceMax= 0;
                for k = 1:dims(3)
                    tmp             = mask(:,:,k);
                    NvolSliceMax    = max(NvolSliceMax,numel(tmp(tmp>0)));
                end
                maxMemoryPerSlice = memoryDynamicPerVoxel * NvolSliceMax;
                maxSlice = floor((maxMemory - memoryRequiredFix)/maxMemoryPerSlice);
                NSegment = ceil(dims(3)/maxSlice);
            end

            fprintf('Data is divided into %d segments\n',NSegment);
        end

        % This function create a full out structure variable if the data is divided into multiple segments
        function out = restore_segment_structure(out,out_tmp,slice,ksegment)
        % Input
        % ---------
        % out       : askadam out structure final output 
        % out_tmp   : temporary out structure of each segment
        % slice     : slices where the segment belongs to
        % ksegment  : current segment number
        % 
            % reformat out structure
            fn1 = fieldnames(out_tmp);
            for kfn1 = 1:numel(fn1)
                fn2 = fieldnames(out_tmp.(fn1{kfn1}));
                for kfn2 = 1:numel(fn2)
                    if isscalar(out_tmp.(fn1{kfn1}).(fn2{kfn2})) % scalar value
                        out.(fn1{kfn1}).(fn2{kfn2})(ksegment) = out_tmp.(fn1{kfn1}).(fn2{kfn2});
                    else
                        % image result
                        out.(fn1{kfn1}).(fn2{kfn2})(:,:,slice) = out_tmp.(fn1{kfn1}).(fn2{kfn2});
                    end
                        
                end
            end
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

    end

end