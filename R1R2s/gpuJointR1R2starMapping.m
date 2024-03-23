classdef gpuJointR1R2starMapping
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% MCR-MWI and MCR-DIMWI models
% Date created: 21 Jan 2024 (need further testing)
% Date modified:

    properties (Constant)
            gyro = 42.57747892;
    end

    properties (GetAccess = public, SetAccess = protected)
        
        te;
        tr;
        fa;
        % epgx_params;

    end
    
    methods
        %% Constructor
        function obj = gpuJointR1R2starMapping(te,tr,fa)
            obj.te = double(te(:));
            obj.tr = double(tr(:));
            obj.fa = double(fa(:));

        end

        %% Data fitting
        % prepare data for MCR-MWI fitting
        function [algoPara,data_obj] = data_preparation(obj,algoPara,imgPara)

            disp('===============================================');
            disp('Myelin water imaing: MCR-DIMWI Data Preparation');
            disp('===============================================');

            %%%%%%%%%% validate algorithm and image parameters %%%%%%%%%%
            [algoPara,imgPara] = obj.check_and_set_default(algoPara,imgPara);

            %%%%%%%%%% capture all image parameters %%%%%%%%%%
            data  = double(imgPara.img);
            b1map = double(imgPara.b1map);
            mask  = imgPara.mask>0;
            
            if algoPara.isNormData
                [scaleFactor, data] = mwi_image_normalisation(data, mask);
            else
                scaleFactor = 1;
            end

            %%%%%%%%%% check advanced starting point strategy %%%%%%%%%%
            % use lowest flip angle to estimate R2*
            r2s0            = obj.R2star_trapezoidal(mean(abs(data),5),obj.te);
            mask_valida_r2s = and(~isnan(r2s0),~isinf(r2s0));
            r2s0(mask_valida_r2s == 0) = 0;

            despot1_obj     = despot1(obj.tr,obj.fa);
            [t10, m00]    = despot1_obj.estimate(permute(abs(data(:,:,:,1,:)),[1 2 3 5 4]),mask,b1map);
            r10 = 1./t10;

            % smooth out outliers
            m00    = medfilt3(m00);
            r10    = medfilt3(r10);
            r2s0   = medfilt3(r2s0);
            
            %%%%%%%%%% Batch processing preparation %%%%%%%%%%
            % set up data_obj for batch processing
            % input data
            data_obj.data   = data;
            data_obj.b1map  = b1map;
            data_obj.m00    = m00;
            data_obj.r10    = r10;
            data_obj.r2s0   = r2s0;
            data_obj.mask   = mask;
            data_obj.scaleFactor = scaleFactor;

            disp('Data preparation step is completed.');

        end

        % Data fitting on whole dataset
        function res_obj = estimate(obj,algoPara,data_obj)
            gpuDevice([]);
            gpu = gpuDevice;

            disp('=============================================================');
            disp('Myelin water imaing: MCR-DIMWI model fitting - askAdam solver');
            disp('=============================================================');

            % display some messages
            obj.display_algorithm_info(algoPara);
    
            %%%%%%%%%% create directory for temporary results %%%%%%%%%%
            % default_output_filename = 'mwi_mcr_results.mat';
            % [output_dir, output_filename]                = obj.setup_output(algoPara,default_output_filename);
            % identifier = algoPara.identifier;

            %%%%%%%%%% log command window display to a text file %%%%%%%%%%
            % logFilename = fullfile(output_dir, ['run_mwi_' identifier '.log']);
            % logFilename = check_unique_filename_seqeunce(logFilename);
            % diary(logFilename)

            % fprintf('Output directory                : %s\n',output_dir);
            % fprintf('Intermediate results identifier : %s\n',identifier);

            % main
            try
                
                %%%%%%%%%% progress display %%%%%%%%%%

                disp('--------')
                disp('Progress')
                disp('--------')
                res_obj = obj.fit(data_obj,algoPara);

                %%%%%%%%%% fitting main %%%%%%%%%%
                
                % save(output_filename,'res_obj','-append');

                % disp('The process is completed.')

            catch ME
    
                % % close log file
                % disp('There was an error! Please check the command window/error message file for more information.');
                % diary off
                % 
                % % open a new text file for error message
                % errorMessageFilename = fullfile(output_dir, ['run_mwi_' identifier '.error']);
                % errorMessageFilename = check_unique_filename_seqeunce(errorMessageFilename);
                % fid = fopen(errorMessageFilename,'w');
                % fprintf(fid,'The identifier was:\n%s\n\n',ME.identifier);
                % fprintf(fid,'The message was:\n\n');
                % msgString = getReport(ME,'extended','hyperlinks','off');
                % fprintf(fid,'%s',msgString);
                % fclose(fid);
                % 
                % rethrow the error message to command window
                rethrow(ME);
            end

            % clear GPU
            if gpuDeviceCount > 0
                gpuDevice([]);
            end

        end

        % Data fitting on 1 batch
        function fitRes = fit(obj, data_obj, algoPara)
        %
        % Input
        % -----------
        % data_obj  : data object generated by data_preparation function
        % algoPara  : fitting algorithm parameters
        %   .maxIter            : no. of maximum iterations, default = 4000
        %   .initialLearnRate   : initial gradient step size, defaulr = 0.01
        %   .decayRate          : decay rate of gradient step size; learningRate = initialLearnRate / (1+decayRate*epoch), default = 0.0005
        %   .convergenceValue   : convergence tolerance, based on the slope of last 'convergenceWindow' data points on loss, default = 1e-8
        %   .convergenceWindow  : number of data points to check convergence, default = 20
        %   .tol                : stop criteria on metric value, default = 1e-3 for L1)
        %   .lambda             : regularisation parameter, default = 0 (no regularisation)
        %   .TVmode             : mode for TV regulariation, '2D'|'3D', default = '2D'
        %   .lossFunction       : loss for data fidelity term, 'L1'|'L2'|'MSE', default = 'L1'
        %   .isdisplay          : online display the fitting process on figure, true|false, defualt = false
        %   .isWeighted         : if weights the loss, true:false, default = true
        %   .weightMethod       : weighting method for loss, 'norm'|'1stecho', default = '1stecho'
        %   .weightPower        : power for weighting method '1stecho', default = 2;
        %   .isNormData         : is normalised data for data fitting, default = true;
        % pars0     : 4D parameter starting points of fitting, [x,y,slice,param], 4th dimension corresponding to fitting  parameters with order [fa,Da,De,ra,p2] (optional)
        % 
        % Output
        % -----------
        % out       : output structure
        %   .final      : final results
        %       .fa         : Intraneurite volume fraction
        %       .loss       : final loss metric
        %   .min        : results with the minimum loss metric across all iterations
        %       .fa         : Intraneurite volume fraction
        %       .loss       : loss metric      
        % fa        : final Intraneurite volume fraction
        %
        % Description: askAdam Image-based NEXI model fitting
        %
        % Kwok-Shing Chan @ MGH
        % kchan2@mgh.harvard.edu
        % Date created: 8 Dec 2023
        % Date modified:
        %
        %
            
            % % check GPU
            % gpuDevice;
            
            % check image size
            dims    = size(data_obj.data);

            if isfield(data_obj,'mask')
                % assume mask is 3D
                mask = repmat(data_obj.mask,1,1,1,numel(obj.te),numel(obj.fa));
            else
                % if no mask input then fit everthing
                mask = ones(dims);
            end
            Nsample = numel(mask(mask>0))/numel(obj.te)/numel(obj.fa);

            % put data input gpuArray
            mask        = gpuArray(logical(mask));  
            img         = gpuArray(single(abs(data_obj.data)));
            rho_max     = prctile(data_obj.m00(:),99);
            true_famp   = data_obj.b1map .* permute(deg2rad(obj.fa),[2 3 4 5 1]);
            
            % set fitting boundary
            %    [M0,       R2s,R1]
            ub = [rho_max*2,  200,   2.5];
            lb = [0,          0.5,   0.1];

            parameters = obj.initialise_model(data_obj,ub,lb);   % all parameters are betwwen [0,1]
            parameters = obj.set_boundary(parameters);

            % weights
            if algoPara.isWeighted
                switch lower(algoPara.weightMethod)
                    case 'norm'
                        % weights using echo intensity, as suggested in Nam's paper
                        w = sqrt(abs(img));
                    case '1stecho'
                        p = algoPara.weightPower;
                        % weights using the 1st echo intensity of each flip angle
                        w       = bsxfun(@rdivide,abs(img).^p,abs(img(:,:,:,1,:)).^p);
                        w(w>1)  = 1;
                end
            else
                % compute the cost without weights
                w = ones(size(img));
            end
            w = w(mask>0);
            w = dlarray(gpuArray(w).','CB');

            % display optimisation algorithm parameters
            disp(['Maximum no. of iteration = ' num2str(algoPara.maxIter)]);
            disp(['Loss function            = ' algoPara.lossFunction]);
            disp(['Loss tolerance           = ' num2str(algoPara.tol)]);
            disp(['Convergence tolerance    = ' num2str(algoPara.convergenceValue)]);
            disp(['Initial learning rate    = ' num2str(algoPara.initialLearnRate)]);
            disp(['Learning rate decay rate = ' num2str(algoPara.decayRate)]);
            if algoPara.lambda > 0 
                disp(['Regularisation parameter = ' num2str(algoPara.lambda)]);
            end
            
            % clear cache before running everthing
            accfun = dlaccelerate(@obj.modelGradients);
            clearCache(accfun)

            % optimisation process
            averageGrad     = [];
            averageSqGrad   = [];
            
            if algoPara.isdisplay
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
            convergenceCurr         = 1+algoPara.convergenceValue;
            convergenceBuffer       = ones(algoPara.convergenceWindow,1);
            A                       = [(1:algoPara.convergenceWindow).', ones(algoPara.convergenceWindow,1)]; % A matrix to derive convergence
            % optimisation
            for epoch = 1:algoPara.maxIter
                
                % make sure the parameters are [0,1]
                parameters = obj.set_boundary(parameters);

                % Evaluate the model gradients and loss using dlfeval and the modelGradients function.
                [gradients,loss,loss_fidelity,loss_reg] = dlfeval(accfun,parameters,img,true_famp,mask,w,Nsample,ub,lb,algoPara);
            
                % Update learning rate.
                learningRate = algoPara.initialLearnRate / (1+ algoPara.decayRate*epoch);
                
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
                if convergenceCurr < algoPara.convergenceValue && epoch >= algoPara.convergenceWindow
                    fprintf('Convergence is less than the tolerance %e \n',algoPara.convergenceValue);
                    break
                end
                if loss < algoPara.tol
                    fprintf('Loss is less than the tolerance %e \n',algoPara.tol);
                    break
                end

                % Update the network parameters using the adamupdate function.
                [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
                    averageSqGrad,epoch,learningRate);
                
                % if display then plot the loss
                if algoPara.isdisplay
                    
                    addpoints(lineLoss,epoch, loss);
                
                    D = duration(0,0,toc(start),'Format','hh:mm:ss');
                    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
                    drawnow
                end
                % every 100 iterations shows details
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
            parameters = obj.set_boundary(parameters);
            
            % rescale the network parameters
            parameters = obj.rescale_parameters(parameters,ub,lb);

            % result at final iteration
            fitRes.final.mask   = gather(mask(:,:,:,1,1))>0; 
            fitRes.final.M0map      = single(gather(extractdata(parameters.m0.* mask(:,:,:,1,1)))) * data_obj.scaleFactor;
            fitRes.final.R2starmap  = single(gather(extractdata(parameters.r2s.* mask(:,:,:,1,1))));
            fitRes.final.R1map      = single(gather(extractdata(parameters.r1.* mask(:,:,:,1,1))));
            fitRes.final.loss = loss;
            fitRes.final.loss_fidelity = double(gather(extractdata(loss_fidelity)));
            if algoPara.lambda == 0
                fitRes.final.loss_reg      = 0;
            else
                fitRes.final.loss_reg      = double(gather(extractdata(loss_reg)));
            end

            % result at minimum loss
            % rescale the network parameters
            parameters_minLoss      = obj.rescale_parameters(parameters_minLoss,ub,lb);
            fitRes.min.mask   = gather(mask(:,:,:,1,1))>0; 
            fitRes.min.M0map      = single(gather(extractdata(parameters_minLoss.m0.* mask(:,:,:,1,1)))) * data_obj.scaleFactor;
            fitRes.min.R2starmap      = single(gather(extractdata(parameters_minLoss.r2s.* mask(:,:,:,1,1))));
            fitRes.min.R1map      = single(gather(extractdata(parameters_minLoss.r1.* mask(:,:,:,1,1))));
           
            fitRes.min.loss             = minLoss;
            fitRes.min.loss_fidelity    = double(gather(extractdata(minLossFidelity)));
            if algoPara.lambda == 0
                fitRes.min.loss_reg     = 0;
            else
                fitRes.min.loss_reg     = double(gather(extractdata(minLossRegularisation)));
            end

            disp('The processing is completed.')

        end

        % compute the gradient and loss of forward modelling
        function [gradients,loss,loss_fidelity,loss_reg] = modelGradients(obj, parameters, y, true_famp, mask, weights, Nsample, ub, lb,fitParams)
            
           % rescale network parameter to true values
            parameters_true = obj.rescale_parameters(parameters,ub,lb);

            % Forward model
            Shat = obj.FWD(parameters_true,true_famp);
            Shat(isinf(Shat))   = 0;
            Shat(isnan(Shat))   = 0;

            % Masking
            % magnitude data
            Shat    = dlarray(Shat(mask>0).',   'CB');
            y       = dlarray(abs(y(mask>0)).', 'CB');

            % Data fidelity term
            switch lower(fitParams.lossFunction)
                case 'l1'
                    loss_fidelity = l1loss(Shat, y, weights);
                case 'l2'
                    loss_fidelity = l2loss(Shat, y, weights);
                case 'mse'
                    loss_fidelity = mse(Shat, y);
                case 'huber'
                    loss_fidelity = huber(Shat, y, weights);
            end
            
            % regularisation term
            if fitParams.lambda > 0
                cost        = obj.reg_TV(squeeze(parameters.M0),mask(:,:,:,1,1),fitParams.TVmode,fitParams.voxelSize);
                loss_reg    = sum(abs(cost),"all")/Nsample *fitParams.lambda;
            else
                loss_reg = 0;
            end
            
            % compute loss
            loss = loss_fidelity + loss_reg;
            
            % Calculate gradients with respect to the learnable parameters.
            gradients = dlgradient(loss,parameters);
        
        end
        
        % MCRMWI signal, compute the forward model
        function signal = FWD(obj, pars, true_famp)
        % Forward model to generate MCRMWI signal
            TE  = permute(obj.te,[2 3 4 1 5]);
            
            signal = obj.model_jointR1R2s(pars.m0,pars.r2s,pars.r1,TE,obj.tr,true_famp);
        end
        
        %% Signal models
        % Total variation regularisation
        function cost = reg_TV(obj,img,mask,TVmode,voxelSize)
            % voxel_size = [1 1 1];
            % Vr      = 1./sqrt(abs(mask.*obj.gradient_operator(img,voxel_size)).^2+eps);
            cost = sum(abs(mask.*obj.gradient_operator(img,voxelSize,TVmode)),4);

            % cost    = obj.divergence_operator(mask.*(Vr.*(mask.*obj.gradient_operator(img,voxel_size))),voxel_size);
        end
        
        %% Utility functions
        % check and set default
        function [algoPara2,imgPara2] = check_and_set_default(obj,algoPara,imgPara)
        
            imgPara2    = imgPara;
            algoPara2   = algoPara;
            
            %%%%%%%%%% 1. check algorithm parameters %%%%%%%%%%
            % check debug
            try algoPara2.DEBUG             = algoPara.DEBUG;         	catch; algoPara2.DEBUG = false; end
            % check maximum iterations allowed
            try algoPara2.maxIter           = algoPara.maxIter;     	catch; algoPara2.maxIter = 4000; end
            % check function tolerance
            try algoPara2.tol               = algoPara.tol;      	    catch; algoPara2.tol = 1e-3; end
            % check step tolerance
            try algoPara2.convergenceValue  = algoPara.convergenceValue;catch; algoPara2.convergenceValue = 1e-8; end
            % fast fitting when EPG enabled
            try algoPara2.convergenceWindow = algoPara.convergenceWindow;catch; algoPara2.convergenceWindow = 20; end
            try algoPara2.lossFunction      = algoPara.lossFunction;    catch; algoPara2.lossFunction = 'L1'; end
            % check normalised data before fitting
            try algoPara2.isNormData        = algoPara.isNormData;  	catch; algoPara2.isNormData = true; end
            % check weighted sum of cost function
            try algoPara2.isWeighted        = algoPara.isWeighted;  	catch; algoPara2.isWeighted = true; end
            % check weighted sum of cost function
            try algoPara2.weightPower       = algoPara.weightPower;  	catch; algoPara2.weightPower = 2; end
            
            % leanring rate
            try algoPara2.initialLearnRate  = algoPara.initialLearnRate;catch; algoPara2.initialLearnRate = 0.001; end
            % decay rate
            try algoPara2.decayRate         = algoPara.decayRate;       catch; algoPara2.decayRate = 0.0005; end
            try algoPara2.lambda            = algoPara.lambda;          catch; algoPara2.lambda = 0; end
            try algoPara2.isdisplay         = algoPara.isdisplay;       catch; algoPara2.isdisplay = 0;end
            try algoPara2.lambda            = algoPara.lambda;          catch; algoPara2.lambda = 0;end
            try algoPara2.TVmode            = algoPara.TVmode;          catch; algoPara2.TVmode = '2D';end
            try algoPara2.voxelSize         = algoPara.voxelSize;       catch; algoPara2.voxelSize = [2,2,2];end
            
            %%%%%%%%%% 2. check data integrity %%%%%%%%%%
            disp('-----------------------');
            disp('Checking data integrity');
            disp('-----------------------');
            % check if the number of echo times matches with the data
            if numel(obj.te) ~= size(imgPara.img,4)
                error('The length of TE does not match with the 4th dimension of the image.');
            end
            if numel(obj.fa) ~= size(imgPara.img,5)
                error('The length of flip angle does not match with the 5th dimension of the image.');
            end
            % check signal mask
            try
                imgPara2.mask = imgPara.mask;
                disp('Mask input                : True');
            catch
                imgPara2.mask = max(max(abs(imgPara.img),[],4),[],5)./max(abs(imgPara.img(:))) > 0.05;
                disp('Mask input                : false');
                disp('Default masking method is used.');
            end
          
            disp('Input data is valid.')

        end
        
        % display fitting algorithm information
        function display_algorithm_info(obj,algoPara)
        
            %%%%%%%%%% 3. display some algorithm parameters %%%%%%%%%%
            disp('--------------');
            disp('Fitting option');
            disp('--------------');
            if algoPara.isNormData
                disp('GRE data is normalised before fitting');
            else
                disp('GRE data is not normalised before fitting');
            end
            fprintf('Max. iterations    : %i\n',algoPara.maxIter);
            fprintf('Loss tolerance     : %.2e\n',algoPara.tol);

            disp('Cost function options:');
            if algoPara.isWeighted
                disp('Cost function weighted by echo intensity: True');
                disp(['Weighting method: ' algoPara.weightMethod]);
                if strcmpi(algoPara.weightMethod,'1stEcho')
                    disp(['Weighting power: ' num2str(algoPara.weightPower)]);
                end
            else
                disp('Cost function weighted by echo intensity: False');
            end
            
            
        end

    end

    methods(Static)

        function signal = model_jointR1R2s(m0,r2s,r1,te,TR,fa)
        % f : myelin volume fraction
        % R1f : R1 of free (IE) water
        % kfr : exchange rate from free (IE) water to myelin
        % true_famp: true flip angle map in radian (5D), flip angle in 5th dim
        % ss_pool: 1st-3rd dim:image; 4th dim: TE;5th dim: FA;6th dim: [S_M,S_IEW]

            signal = m0 .* sin(fa) .* (1 - exp(-TR.*r1)) ./ (1 - cos(fa) .* exp(-TR.*r1)) .* ...
                            exp(-te .* r2s);
        
        
        end
        
        function [R2star,S0] = R2star_trapezoidal(img,te)
            % disgard phase information
            img = double(abs(img));
            te = double(te);
            
            dims = size(img);
            
            %% main
            % Trapezoidal approximation of integration
            temp=0;
            for k=1:dims(4)-1
                temp=temp+0.5*(img(:,:,:,k)+img(:,:,:,k+1))*(te(k+1)-te(k));
            end
            
            % very fast estimation
            t2s=temp./(img(:,:,:,1)-img(:,:,:,end));
                
            R2star = 1./t2s;

            S0 = img(1:(numel(img)/dims(end)))'.*exp(R2star(:)*te(1));
            if numel(S0) ~=1
                S0 = reshape(S0,dims(1:end-1));
            end
        end

        % initialise network parameters
        function parameters = initialise_model(data,ub,lb)

            % m00    = min(max(data.m00 ,   lb(1)),ub(1));
            % r2s0   = min(max(data.r2s0,   lb(2)),ub(2));
            % r10    = min(max(data.r10,    lb(3)),ub(3));
            % 
            % m00     = (m00     -lb(1)) / (ub(1)-lb(1));
            % r2s0    = (r2s0    -lb(2)) / (ub(2)-lb(2));
            % r10     = (r10      -lb(3)) / (ub(3)-lb(3));
            m00     = rand(size(data.r2s0),'single');
            r2s0    = rand(size(data.r2s0),'single');
            r10     = rand(size(data.r2s0),'single');
            
            % assume comparment a has long T1 and T2*
            parameters.m0   = gpuArray( dlarray(m00));
            parameters.r2s  = gpuArray( dlarray(r2s0));
            parameters.r1   = gpuArray( dlarray(r10)) ;
            
        end

        % make sure all network parameters stay between 0 and 1
        function parameters = set_boundary(parameters)

            field = fieldnames(parameters);
            for k = 1:numel(field)
                parameters.(field{k})   = max(parameters.(field{k}),0); % Lower bound     
                parameters.(field{k})   = min(parameters.(field{k}),1); % upper bound

            end

        end
        
        % rescale the network parameters between the defined lower/upper bounds
        function parameters = rescale_parameters(parameters,ub,lb)

            parameters.m0   = parameters.m0  * (ub(1)-lb(1)) + lb(1);
            parameters.r2s  = parameters.r2s * (ub(2)-lb(2)) + lb(2);
            parameters.r1   = parameters.r1  * (ub(3)-lb(3)) + lb(3);
            

        end

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
    
        % setup output related operations
        function [output_dir,output_filename] = setup_output(imgPara,output_filename)
        %
        % Input
        % --------------
        % imgPara       : structure array contains all image data
        % Output setting
        % ==============
        %   .output_dir      : directory to store final results (default:
        %                      '/current_directory/mwi_results/')
        %   .output_filename : output filename in text string (default:
        %                      'mwi_results.mat')
        %   .identifier      : temporary file identifier, a 8-digit code (optional)
        %
        % Output
        % --------------
        % output_filename : full output filename with extension
        % temp_filename   : full temporary filename
        % identifier      : 8-digit code (in string)
        %
        % Description: setup output related operations
        %
        % Kwok-shing Chan @ DCCN
        % k.chan@donders.ru.nl
        % Date created: 16 Nov 2020
        % Date modified:
        %
        %
            % check if user specified output directory
            if isfield(imgPara,'output_dir')
                output_dir = imgPara.output_dir;
            else
                output_dir    = fullfile(pwd,'mwi_results');
            end
            if ~exist(output_dir,'dir')
                mkdir(output_dir)
            end
            
            % check if user specified output filename
            if isfield(imgPara,'output_filename')
                output_filename = imgPara.output_filename;
                [~,output_filename,~] = fileparts(output_filename);
                output_filename = [output_filename '.mat'];
            end
            output_filename = fullfile(output_dir,output_filename);
            
        end

        
    end

end