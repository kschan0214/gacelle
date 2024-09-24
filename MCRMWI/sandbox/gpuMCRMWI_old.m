classdef gpuMCRMWI
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% No support for DIMWI yet
% Date created: 21 Jan 2024 (v0.0.2)
% Date modified:

    properties (Constant)
            gyro = 42.57747892;
    end

    properties (GetAccess = public, SetAccess = protected)
        
        % tissue parameters
        x_i     = -0.1;     % ppm
        x_a     = -0.1;     % ppm
        E       = 0.02;     % ppm
        rho_mw  = 0.42;     % relative water proton density relative to IEW
        t1_mw   = 234e-3;   % s
        % hardware setting
        B0      = 3; % T
        B0dir   = [0;0;1]; % main magnetic field direction with respect to FOV
        te;
        tr;
        fa;
        % epgx_params;

    end
    
    methods
        %% Constructor
        function obj = gpuMCRMWI(te,tr,fa,fixed_params)
            obj.te = double(te(:));
            obj.tr = double(tr(:));
            obj.fa = double(fa(:));

            % fixed tissue and scanner parameters
            if nargin == 4
                if isfield(fixed_params,'rho_mw')
                    obj.rho_mw  = double(fixed_params.rho_mw);
                end
                if isfield(fixed_params,'B0')
                    obj.B0      = double(fixed_params.B0);
                end
                if isfield(fixed_params,'B0dir')
                    obj.B0dir   = double(fixed_params.B0dir);
                end
                if isfield(fixed_params,'t1_mw')
                    obj.t1_mw   = double(fixed_params.t1_mw);
                end
            end

        end

        %% Data fitting
        % prepare data for MCR-MWI fitting
        function [algoPara,data_obj_all] = data_preparation(obj,algoPara,imgPara)

            disp('=================================================');
            disp('Myelin water imaing: MCR-(DI)MWI Data Preparation');
            disp('=================================================');

            %%%%%%%%%% create directory for temporary results %%%%%%%%%%
            default_output_filename = 'mcrmwi_results.mat';
            temp_prefix             = 'temp_mcrmwi_';
            [output_dir, output_filename]       = obj.setup_output(imgPara,default_output_filename);
            [temp_dir,temp_prefix, identifier]  = obj.setup_temp_dir(imgPara,temp_prefix);

            %%%%%%%%%% validate algorithm and image parameters %%%%%%%%%%
            [algoPara,imgPara] = obj.check_and_set_default(algoPara,imgPara);

            %%%%%%%%%% capture all image parameters %%%%%%%%%%
            data  = double(imgPara.img);
            b1map = double(imgPara.b1map);
            mask  = imgPara.mask>0;
            fm0   = double(imgPara.fieldmap);
            pini0 = double(imgPara.pini);
            
            if algoPara.isNormData
                [scaleFactor, data] = mwi_image_normalisation(data, mask);
            else
                scaleFactor = 1;
            end

            %%%%%%%%%% check advanced starting point strategy %%%%%%%%%%
            % use lowest flip angle to estimate R2*
            [~,ind]         = min(obj.fa);
            r2s0            = R2star_trapezoidal(data(:,:,:,:,ind),obj.te);
            mask_valida_r2s = and(~isnan(r2s0),~isinf(r2s0));
            r2s0(mask_valida_r2s == 0) = 0;

            % only works for 3T data
            advancedStarting = algoPara.advancedStarting;
            if strcmpi(advancedStarting,'default') || strcmpi(advancedStarting,'robust')
                fprintf('Estimate starting points using predefined model...')
                
                if obj.B0 > 2.5 && obj.B0 < 3.5 % 3T
                    t2s_pre = [10e-3,60e-3];    % [T2sMW, T2sIEW] in second
                    t1_pre  = [234e-3, 1];    	% [T1MW, IEW], in second
                elseif obj.B0 > 1 && obj.B0 < 2 % 1.5T
                    t2s_pre = [10e-3,80e-3];    
                    t1_pre  = [234e-3, 0.7];    
                elseif obj.B0 > 6 && obj.B0 < 8 % 7T
                    t2s_pre = [10e-3,40e-3];    
                    t1_pre  = [234e-3, 1.5];    
                end
                
                switch advancedStarting
                    case 'default'
                        [m00,mwf0,t2siew0,t1iew0] = superfast_mwi_2m_mcr_self(data,obj.te,obj.fa,obj.tr,t2s_pre,t1_pre(1),mask,b1map,'superfast');
                        r1iew0 = 1./t1iew0;
                    case 'robust'
                        [m00,mwf0] = superfast_mwi_2m_mcr(data,obj.te,obj.fa,obj.tr,t2s_pre,t1_pre,mask,b1map);
                end
                m00 = sum(m00,4); % total water
                % also masked out problematic voxels detected by superfast method
                mask_valid_m0 = m00 > 0;
                disp('Completed.')
            end
            % Simple DESPOT1 T1 mapping
            if ~exist('r1iew0','var')
                fprintf('Estimate T1 using DESPOT1...')
                despot1_obj     = despot1(obj.tr,obj.fa);
                [t1iew0, m0]    = despot1_obj.estimate(permute(data(:,:,:,1,:),[1 2 3 5 4]),mask,b1map);
                r1iew0 = 1./t1iew0;
                if ~exist('m00','var')
                    m00 = m0;
                    % also masked out DESPOT1 problematic voxels
                    mask_valid_m0 = m00 > 0;
                end
                mask_valid_r1iew0 = and(r1iew0>0,~isinf(r1iew0));
                r1iew0(mask_valid_r1iew0==0) = 0;
                disp('Completed.')
            end

            mask_nonzero = min(data(:,:,:,1,:) ~= 0,[],5);
            
            % final mask for fitting
            mask = and(and(and(and(mask,mask_valida_r2s),mask_valid_m0),mask_valid_r1iew0),mask_nonzero);

            % smooth out outliers
            m00     = medfilt3(m00);
            r1iew0  = medfilt3(r1iew0);
            r2s0    = medfilt3(r2s0);
            
            %%%%%%%%%% Batch processing preparation %%%%%%%%%%
            % set up data_obj for batch processing
            % input data
            data_obj_all = obj.setup_batch_create_data_obj_slice(data,      mask, 'data');
            data_obj_all = obj.setup_batch_create_data_obj_slice(b1map,     mask, 'b1map',  data_obj_all);
            data_obj_all = obj.setup_batch_create_data_obj_slice(fm0,       mask, 'fm0',    data_obj_all);
            data_obj_all = obj.setup_batch_create_data_obj_slice(pini0,     mask, 'pini0',  data_obj_all);
            % derived initial guess
            data_obj_all = obj.setup_batch_create_data_obj_slice(m00,       mask, 'm00',	data_obj_all);
            data_obj_all = obj.setup_batch_create_data_obj_slice(r1iew0,    mask, 'r1iew0',	data_obj_all);
            data_obj_all = obj.setup_batch_create_data_obj_slice(r2s0,      mask, 'r2s0',	data_obj_all);
            if exist('mwf0','var');     data_obj_all = obj.setup_batch_create_data_obj_slice(mwf0,      mask, 'mwf0',       data_obj_all); end
            if exist('t2siew0','var');  data_obj_all = obj.setup_batch_create_data_obj_slice(t2siew0,   mask, 't2siew0',	data_obj_all); end
            
            m00_max = max(prctile(m00(mask>0),99),iqr(m00(mask>0))*3 + median(m00(mask>0)));
            
            NTotalSamples = numel(mask(mask>0));
            for kbat = 1:numel(data_obj_all)
                data_obj_all(kbat).m00_max = m00_max;
                data_obj            = data_obj_all(kbat);
                batchNumber         = kbat;
                NSamples            = numel(data_obj.mask(data_obj.mask>0));
                save(fullfile(temp_dir,strcat(temp_prefix,'_batch',num2str(kbat))),'data_obj','identifier','scaleFactor','batchNumber',...
                    'NTotalSamples','NSamples');
            end
            algoPara.identifier         = identifier;
            algoPara.temp_dir           = temp_dir;
            algoPara.temp_prefix        = temp_prefix;
            algoPara.output_dir         = output_dir;
            algoPara.output_filename    = output_filename;

            save(output_filename,'algoPara');

            disp('Note that the final mask could be different from the input mask due to data validity.');
            disp('Data preparation step is completed.');

        end

        % Data fitting on whole dataset
        function res_obj = estimate(obj,algoPara,data_obj)
            gpuDevice([]);
            gpu = gpuDevice;

            disp('===============================================================');
            disp('Myelin water imaing: MCR-(DI)MWI model fitting - askAdam solver');
            disp('===============================================================');

            % display some messages
            obj.display_algorithm_info(algoPara);
    
            % determine if the data is provided directly or saved to the disk
            if nargin < 3
                % if no data input then get identifier from algoPara
                data_obj    = [];
                temp_files  = fullfile(algoPara.temp_dir,algoPara.temp_prefix);
                filelist    = dir([temp_files '*']);
                if ~isempty(filelist)
                    Nbatch = numel(filelist);
                end

                isBatch = true;

            elseif ~isempty(data_obj)
                isBatch = false;
                Nbatch  = 1;
            end

            %%%%%%%%%% create directory for temporary results %%%%%%%%%%
            default_output_filename = 'mwi_mcr_results.mat';
            [output_dir, output_filename]                = obj.setup_output(algoPara,default_output_filename);
            identifier = algoPara.identifier;

            %%%%%%%%%% log command window display to a text file %%%%%%%%%%
            logFilename = fullfile(output_dir, ['run_mwi_' identifier '.log']);
            logFilename = check_unique_filename_seqeunce(logFilename);
            diary(logFilename)

            fprintf('Output directory                : %s\n',output_dir);
            fprintf('Intermediate results identifier : %s\n',identifier);

            load('MLP_EPGX_leakyrelu_LeeANN4MWImodel_epoch_100.mat');
            dlnet.alpha = 0.01;
            % main
            try
                
                %%%%%%%%%% progress display %%%%%%%%%%

                disp('--------')
                disp('Progress')
                disp('--------')

                %%%%%%%%%% fitting main %%%%%%%%%%
                for kbat = 1:Nbatch
                    if isBatch
                        % if batch mode then check if the input mat contains output variable
                        temp_mat = strcat(temp_files,['_batch' num2str(kbat)]);
                        variableInfo = who('-file', temp_mat);
                        isFit = ~ismember('res_obj', variableInfo);
                    else
                        NSamples    = numel(data_obj.mask(data_obj.mask>0));
                        isFit       = true;
                    end

                    % display current batch number
                    fprintf('#Slice %3d/%d',kbat,Nbatch);

                    if isFit 
                        if isBatch
                            % if batch mode then load data from disk
                            load(temp_mat,'data_obj','NSamples');
                        end

                        % display number of voxels to be fitted
                        fprintf(',   #Voxel = %6d\n',NSamples);

                        % process non-empty slices
                        if ~data_obj.isEmptySlice

                            res_obj = obj.fit(data_obj,algoPara,dlnet);

                            % save temporary result
                            if isBatch
                                save(temp_mat,'res_obj','-append')
                            end

                        end
                    else
                        if isBatch
                            % if batch mode then load data from disk
                            load(temp_mat,'NSamples','res_obj');
                            % display number of voxels to be fitted
                            fprintf(',   #Voxel = %6d,    this batch is previously processed.\n',NSamples);
                        end
                    end
                end

                % concatenate the result if in batch mode
                if isBatch
                    
                    res_obj = []; isSetup = false;
                    for kbat = 1:Nbatch
                        temp_mat = strcat(temp_files,['_batch' num2str(kbat)]);
                        temp = load(temp_mat);
                        if isfield(temp,'res_obj')
                            if ~isSetup
    
                                fieldname = fieldnames(temp.res_obj);
                                for kfield = 1:numel(fieldname)
                                    
                                    fieldname2 = fieldnames(temp.res_obj.(fieldname{kfield}));

                                    for kfield2 = 1:numel(fieldname2)
                                        if numel(temp.res_obj.(fieldname{kfield}).(fieldname2{kfield2})) ~= 1
                                            dims = [size(temp.res_obj.(fieldname{kfield}).(fieldname2{kfield2}),1),size(temp.res_obj.(fieldname{kfield}).(fieldname2{kfield2}),2),...
                                                Nbatch,size(temp.res_obj.(fieldname{kfield}).(fieldname2{kfield2}),3)];
                                            res_obj.(fieldname{kfield}).(fieldname2{kfield2}) = zeros(dims);
                                        else
                                            res_obj.(fieldname{kfield}).(fieldname2{kfield2}) = zeros(1,Nbatch);
                                        end
                                    end
                                end
                                isSetup = true;
                            end
                            fieldname = fieldnames(res_obj);
                            for kfield = 1:numel(fieldname)
                                fieldname2 = fieldnames(res_obj.(fieldname{kfield}));
                                for kfield2 = 1:numel(fieldname2)
                                    if numel(temp.res_obj.(fieldname{kfield}).(fieldname2{kfield2})) ~= 1
                                        res_obj.(fieldname{kfield}).(fieldname2{kfield2}) (:,:,kbat,:) = temp.res_obj.(fieldname{kfield}).(fieldname2{kfield2}) ;
                                    else
                                        res_obj.(fieldname{kfield}).(fieldname2{kfield2}) (kbat) = temp.res_obj.(fieldname{kfield}).(fieldname2{kfield2}) ;
                                    end
                                end
                            end
                        end
                    end
                    delete(strcat(temp_files,'_batch*'));
                    rmdir(algoPara.temp_dir);
                end
                save(output_filename,'res_obj','-append');

                disp('The process is completed.')

            catch ME
    
                % close log file
                disp('There was an error! Please check the command window/error message file for more information.');
                diary off
                
                % open a new text file for error message
                errorMessageFilename = fullfile(output_dir, ['run_mwi_' identifier '.error']);
                errorMessageFilename = check_unique_filename_seqeunce(errorMessageFilename);
                fid = fopen(errorMessageFilename,'w');
                fprintf(fid,'The identifier was:\n%s\n\n',ME.identifier);
                fprintf(fid,'The message was:\n\n');
                msgString = getReport(ME,'extended','hyperlinks','off');
                fprintf(fid,'%s',msgString);
                fclose(fid);
                
                % rethrow the error message to command window
                rethrow(ME);
            end

            % clear GPU
            if gpuDeviceCount > 0
                gpuDevice([]);
            end

        end

        % Data fitting on 1 batch
        function fitRes = fit(obj, data_obj, algoPara, dlnet)
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
            img         = gpuArray(single(data_obj.data));
            rho_max     = data_obj.m00_max;
            true_famp   = data_obj.b1map .* permute(deg2rad(obj.fa),[2 3 4 5 1]);
            
            % set fitting boundary
            %    [M0MW,       M0IW,       M0EW,       R2sMW,R2sIW,R2sEW,T1IEW,kIEWM,FreqMW,FreqIW,totalfield,pini]
            ub = [rho_max*1.5,rho_max*1.5,rho_max*1.5,200,  50,   50,   2,    20,   20,    20,    300,       2*pi];
            lb = [eps,        eps,        eps,         50,   2,    2, 0.2,   eps,  -20,   -20,   -300,      -2*pi];

            [parameters,extra_data] = obj.initialise_model(data_obj,algoPara,ub,lb);   % all parameters are betwwen [0,1]
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
                        w = bsxfun(@rdivide,abs(img).^p,abs(img(:,:,:,1,:)).^p);
                end
            else
                % compute the cost without weights
                w = ones(size(img));
            end
            w = w(mask>0);
            if algoPara.isComplex
                w = repmat(w,2,1);
            end
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
                [gradients,loss,loss_fidelity,loss_reg] = dlfeval(accfun,parameters,img,true_famp,mask,w,extra_data,Nsample,ub,lb,algoPara,dlnet);
            
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
            parameters  = obj.rescale_parameters(parameters,algoPara,ub,lb);
            % result at final iteration
            fitRes.final.S0_MW      = single(gather(extractdata(parameters.s0mw.* mask(:,:,:,1,1))));
            fitRes.final.S0_IW      = single(gather(extractdata(parameters.s0iw.* mask(:,:,:,1,1))));
            fitRes.final.S0_EW      = single(gather(extractdata(parameters.s0ew.* mask(:,:,:,1,1))));
            fitRes.final.R2s_MW     = single(gather(extractdata(parameters.r2smw.* mask(:,:,:,1,1))));
            fitRes.final.R2s_IW     = single(gather(extractdata(parameters.r2siw.* mask(:,:,:,1,1))));
            fitRes.final.R2s_EW     = single(gather(extractdata(parameters.r2sew.* mask(:,:,:,1,1))));
            fitRes.final.R1_IEW     = single(gather(extractdata(parameters.r1iew.* mask(:,:,:,1,1))));
            fitRes.final.kiewm      = single(gather(extractdata(parameters.kiewm.* mask(:,:,:,1,1))));
            fitRes.final.Freq_MW    = single(gather(extractdata(parameters.freq_mw.* mask(:,:,:,1,1))));
            fitRes.final.Freq_IW    = single(gather(extractdata(parameters.freq_iw.* mask(:,:,:,1,1))));
            if algoPara.isComplex
                fitRes.final.Freq_BKG   = single(gather(extractdata(parameters.totalField.* mask(:,:,:,1,1))));
                fitRes.final.pini       = single(gather(extractdata(parameters.pini.* mask(:,:,:,1,1))));
            end
            fitRes.final.loss = loss;

            % result at minimum loss
            % rescale the network parameters
            parameters_minLoss      = obj.rescale_parameters(parameters_minLoss,algoPara,ub,lb);
            fitRes.min.S0_MW        = single(gather(extractdata(parameters_minLoss.s0mw.* mask(:,:,:,1,1))));
            fitRes.min.S0_IW        = single(gather(extractdata(parameters_minLoss.s0iw.* mask(:,:,:,1,1))));
            fitRes.min.S0_EW        = single(gather(extractdata(parameters_minLoss.s0ew.* mask(:,:,:,1,1))));
            fitRes.min.R2s_MW       = single(gather(extractdata(parameters_minLoss.r2smw.* mask(:,:,:,1,1))));
            fitRes.min.R2s_IW       = single(gather(extractdata(parameters_minLoss.r2siw.* mask(:,:,:,1,1))));
            fitRes.min.R2s_EW       = single(gather(extractdata(parameters_minLoss.r2sew.* mask(:,:,:,1,1))));
            fitRes.min.R1_IEW       = single(gather(extractdata(parameters_minLoss.r1iew.* mask(:,:,:,1,1))));
            fitRes.min.kiewm        = single(gather(extractdata(parameters_minLoss.kiewm.* mask(:,:,:,1,1))));
            fitRes.min.Freq_MW      = single(gather(extractdata(parameters_minLoss.freq_mw.* mask(:,:,:,1,1))));
            fitRes.min.Freq_IW      = single(gather(extractdata(parameters_minLoss.freq_iw.* mask(:,:,:,1,1))));
            if algoPara.isComplex
                fitRes.min.Freq_BKG   = single(gather(extractdata(parameters_minLoss.totalField.* mask(:,:,:,1,1))));
                fitRes.min.pini       = single(gather(extractdata(parameters_minLoss.pini.* mask(:,:,:,1,1))));
            end
            fitRes.min.loss = loss;

            disp('The processing is completed.')

        end

        % compute the gradient and loss of forward modelling
        function [gradients,loss,loss_fidelity,loss_reg] = modelGradients(obj, parameters, dlR, true_famp, mask, weights, extra_data,Nsample, ub, lb,fitParams,dlnet)
            
           % rescale network parameter to true values
            parameters_true = obj.rescale_parameters(parameters,fitParams,ub,lb);

            if fitParams.isComplex
                totalfield  = parameters_true.totalField;
                pini        = parameters_true.pini;
            else
                totalfield  = extra_data.totalField;
                pini        = extra_data.pini;
            end
            
            % Forward model
            [dlR_real, dlR_imag]        = obj.FWD(parameters_true,totalfield,pini,true_famp,dlnet);
            dlR_real(isinf(dlR_real))   = 0;
            dlR_real(isnan(dlR_real))   = 0;
            dlR_imag(isinf(dlR_imag))   = 0;
            dlR_imag(isnan(dlR_imag))   = 0;

            % Masking
            if fitParams.isComplex
                % complex-valued data
                R   = dlarray(cat(1,dlR_real(mask>0),dlR_imag(mask>0)).',   'CB');
                dlR = dlarray(cat(1,real(dlR(mask>0)),imag(dlR(mask>0))).', 'CB');
            else
                % magnitude data
                R   = dlarray(sqrt(dlR_real(mask>0).^2 + dlR_imag(mask>0).^2).',    'CB');
                dlR = dlarray(abs(dlR(mask>0)).',                                   'CB');
            end

            % Data fidelity term
            switch lower(fitParams.lossFunction)
                case 'l1'
                    loss_fidelity = l1loss(R, dlR, weights);
                case 'l2'
                    loss_fidelity = l2loss(R, dlR, weights);
                case 'mse'
                    loss_fidelity = mse(R, dlR);
            end
            
            % regularisation term
            if fitParams.lambda > 0
                cost        = obj.reg_TV(squeeze(parameters.s0ew),mask(:,:,:,1,1),fitParams.TVmode,fitParams.voxelSize);
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
        function [s_real, s_imag] = FWD(obj, pars, totalField, pini, true_famp, dlnet)
        % Forward model to generate MCRMWI signal
            TE = permute(obj.te,[2 3 4 1 5]);

            scaleFactor = pars.s0iw + pars.s0ew + pars.s0mw / obj.rho_mw;
            fx = (pars.s0mw / obj.rho_mw) ./ scaleFactor;

            Nfa = numel(obj.fa);
            
            ss = zeros(2,numel(fx),Nfa, 'like', pars.r2siw );
            for k = 1:Nfa
                tmp_famp = true_famp(:,:,:,1,k);
                features    = feature_preprocess_LeeANN4MWImodel(fx(:),1./pars.r1iew(:),...
                                                              pars.kiewm(:),tmp_famp(:),obj.tr,obj.t1_mw);
                features    = gpuArray( dlarray(features,'CB'));
                
                ss(:,:,k) = reshape(mlp_model_leakyRelu(dlnet.parameters,features,dlnet.alpha),[2 numel(fx)]);
            end
            ss = permute(reshape( ss,2,size(fx,1),size(fx,2),size(fx,3),Nfa),[2 3 4 6 5 1]).*scaleFactor;

            s_real =  ss(:,:,:,:,:,1).*(pars.s0ew./(pars.s0iw+pars.s0ew)) .* exp(-TE .*  pars.r2sew ) .* ...
                        cos(2.*pi.*totalField.*TE + pini) + ...                                 % EW 
                      ss(:,:,:,:,:,1).*(pars.s0iw./(pars.s0iw+pars.s0ew)) .* exp(-TE .* pars.r2siw) .* ...
                        cos(2.*pi.*pars.freq_iw.*TE + 2.*pi.*totalField.*TE + pini) + ...       % IW
                      ss(:,:,:,:,:,2).*obj.rho_mw .* exp(-TE .* (pars.r2smw)) .* ...
                        cos(2.*pi.*pars.freq_mw.*TE + 2.*pi.*totalField.*TE + pini) ;           % MW

            s_imag =  ss(:,:,:,:,:,1).*(pars.s0ew./(pars.s0iw+pars.s0ew)) .* exp(-TE .*  pars.r2sew ) .* ...
                        sin(2.*pi.*totalField.*TE + pini) + ...                                 % EW 
                      ss(:,:,:,:,:,1).*(pars.s0iw./(pars.s0iw+pars.s0ew)) .* exp(-TE .* pars.r2siw) .* ...
                        sin(2.*pi.*pars.freq_iw.*TE + 2.*pi.*totalField.*TE + pini) + ...       % IW
                      ss(:,:,:,:,:,2).*obj.rho_mw .* exp(-TE .* (pars.r2smw)) .* ...
                        sin(2.*pi.*pars.freq_mw.*TE + 2.*pi.*totalField.*TE + pini) ;           % MW
                
        end
        
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
            % check # of phase-corrupted echoes
            try algoPara2.isComplex         = algoPara.isComplex;       catch; algoPara2.isComplex = ~isreal(imgPara.img); end
            % leanring rate
            try algoPara2.initialLearnRate  = algoPara.initialLearnRate;catch; algoPara2.initialLearnRate = 0.01; end
            % decay rate
            try algoPara2.decayRate         = algoPara.decayRate;       catch; algoPara2.decayRate = 0.0005; end
            try algoPara2.lambda            = algoPara.lambda;          catch; algoPara2.lambda = 0; end
            try algoPara2.isdisplay         = algoPara.isdisplay;       catch; algoPara2.isdisplay = 0;end
            try algoPara2.lambda            = algoPara.lambda;          catch; algoPara2.lambda = 0;end
            try algoPara2.TVmode            = algoPara.TVmode;          catch; algoPara2.TVmode = '2D';end
            try algoPara2.voxelSize         = algoPara.voxelSize;       catch; algoPara2.voxelSize = [2,2,2];end

            % check advanced starting points strategy
            try algoPara2.advancedStarting  = algoPara.advancedStarting;catch; algoPara2.advancedStarting = [];end
            
            
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
            % check field map
            try
                imgPara2.fieldmap = imgPara.fieldmap;
                disp('Field map input           : True');
            catch
                imgPara2.fieldmap = zeros([size(imgPara2.mask) length(imgPara.fa)]);
                disp('Field map input           : False');
            end
            % check initial phase map
            try
                imgPara2.pini = imgPara.pini;
                disp('Initial phase input       : True');
            catch
                imgPara2.pini = nan(size(imgPara2.mask));
                disp('Initial phase input       : False');
            end
            
            try algoPara2.autoSave = algoPara.autoSave; catch; algoPara2.autoSave = true; end
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
            % type of fitting
            if algoPara.isComplex==0
                disp('Fitting with magnitude data');
            else 
                disp('Fitting with complex data');
            end

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
            
            disp('--------------------------');
            disp('Multi-compartment T1 model');
            disp('--------------------------');
            disp('Exchange  - to be fitted');
            disp('T1mw      - fixed');
            
            % disp('-------------------------------')
            % disp('Parameter to be fixed for DIMWI')
            % disp('-------------------------------')
            % disp(['Field strength (T)                       : ' num2str(obj.B0)]);
            % disp(['B0 direction(x,y,z)                      : ' num2str(obj.B0dir(:)')]);
            % disp(['Relative myelin water density            : ' num2str(obj.rho_mw)]);
            % disp(['Myelin isotropic susceptibility (ppm)    : ' num2str(obj.x_i)]);
            % disp(['Myelin anisotropic susceptibility (ppm)  : ' num2str(obj.x_a)]);
            % disp(['Exchange term (ppm)                      : ' num2str(obj.E)]);
            
        end

    end

    methods(Static)
        
        % initialise network parameters
        function [parameters,extra_data] = initialise_model(data_obj,fitParams,ub,lb)

            img_size = size(data_obj.mask);

            data_obj.m00    = min(max(data_obj.m00 ,  lb(1)),ub(1));
            data_obj.r2s0   = min(max(data_obj.r2s0,  lb(5)),ub(5));
            data_obj.r1iew0 = min(max(data_obj.r1iew0,  lb(7)),ub(7));
            data_obj.fm0    = min(max(data_obj.fm0 ,  lb(11)),ub(11));
            data_obj.pini0  = min(max(data_obj.pini0, lb(12)),ub(12));
            
            s0mw        = ((data_obj.m00 * 0.1)             -lb(1)) / (ub(1)-lb(1));
            s0iw        = ((data_obj.m00 * 0.6)             -lb(2)) / (ub(2)-lb(2));
            s0ew        = ((data_obj.m00 * 0.3)             -lb(3)) / (ub(3)-lb(3));
            r2smw       = ((100 + rand(img_size,'single'))  -lb(4)) / (ub(4)-lb(4));
            % r2smw       = (100*ones(img_size,'single')      -lb(4)) / (ub(4)-lb(4));
            r2siw       = ((data_obj.r2s0 - 2.5)            -lb(5)) / (ub(5)-lb(5));
            r2sew       = ((data_obj.r2s0 + 2.5)            -lb(6)) / (ub(6)-lb(6));
            r1iew       = ((data_obj.r1iew0 )               -lb(7)) / (ub(7)-lb(7));
            kiewm       = (rand(img_size,'single')          -lb(8)) / (ub(8)-lb(8));
            % kiewm       = (0*rand(img_size,'single')        -lb(8)) / (ub(8)-lb(8));
            freq_mw     = (5+rand(img_size,'single')        -lb(9)) / (ub(9)-lb(9));
            freq_iw     = (-2+rand(img_size,'single')       -lb(10)) / (ub(10)-lb(10));
            % freq_mw     = (5*ones(img_size,'single')        -lb(9)) / (ub(9)-lb(9));
            % freq_iw     = (-2*ones(img_size,'single')       -lb(10)) / (ub(10)-lb(10));
            totalField  = (data_obj.fm0                     -lb(11)) / (ub(11)-lb(11));
            pini        = (data_obj.pini0                   -lb(12)) / (ub(12)-lb(12));

            % assume comparment a has long T1 and T2*
            parameters.s0iw         = gpuArray( dlarray(s0iw));
            parameters.s0ew         = gpuArray( dlarray(s0ew));
            parameters.s0mw         = gpuArray( dlarray(s0mw));
            parameters.r1iew        = gpuArray( dlarray(r1iew)) ;
            parameters.kiewm        = gpuArray( dlarray(kiewm)) ;
            parameters.r2siw        = gpuArray( dlarray(r2siw)) ;
            parameters.r2sew        = gpuArray( dlarray(r2sew)) ;
            parameters.r2smw        = gpuArray( dlarray(r2smw)) ;
            parameters.freq_mw      = gpuArray( dlarray(freq_mw)) ;
            parameters.freq_iw      = gpuArray( dlarray(freq_iw)) ;
            
            if fitParams.isComplex
                parameters.totalField   = gpuArray( dlarray( permute(totalField,[1 2 3 5 4])) );
                parameters.pini         = gpuArray( dlarray(pini) );
                extra_data              = [];
            else
                extra_data.totalField   = gpuArray( dlarray(permute(totalField,[1 2 3 5 4]) * (ub(11)-lb(11))+lb(11)) );
                extra_data.pini         = gpuArray( dlarray(pini * (ub(12)-lb(12))+lb(12)) );
            end
            
        end
        
        % % check and set default fitting algorithm parameters
        % function fitting2 = check_set_default(fitting)
        %     fitting2 = fitting;
        % 
        %     % get fitting algorithm setting
        %     if ~isfield(fitting,'TVmode')
        %         fitting2.TVmode = '2D';
        %     end
        %     if ~isfield(fitting,'voxelSize')
        %         fitting2.voxelSize = [2,2,2];
        %     end
        % 
        % 
        % end
    
        function data_obj = setup_batch_create_data_obj_slice(data, mask, fieldName, data_obj)
        % data_obj = setup_batch_create_data_obj(data, mask, numBatch, nElement, fieldName, data_obj)
        %
        % Input
        % --------------
        % data          : data to be store in batches, at least 2D, can have singleton dimension
        % mask          : signal mask
        % numBatch      : number of batches the data to be broken down (final)
        % nElement      : number of elements in each batch
        % fieldName     : field name to be store in data_obj, string
        % data_obj      : (optional) previous created data_obj
        % 
        %
        % Output
        % --------------
        % data_obj      : data_obj with data stored
        %
        % Kwok-shing Chan @ DCCN
        % k.chan@donders.ru.nl
        % Date created: 10 August 2021
        % Date modified:
        %
        %

        % initialise data obj
        if nargin < 4 || isempty(data_obj)
            data_obj = [];
        end
        
        % get matrix size of the data
        dim         = size(data);
        if numel(dim) == 1
            dim = [dim, 1, 1];
        elseif numel(dim) == 2
            dim = [dim, 1];
        end
        
        % vectorise image data for all voxels to 1st dimension
        for kz = 1:dim(3)
            img_slice   = data(:,:,kz,:,:);
            mask_slice  = mask(:,:,kz);
        
            % find masked voxels
            ind = find(mask_slice>0);
        
            if ~isempty(ind)
                if nargin < 4
                    data_obj(kz).isEmptySlice = false;
                    data_obj(kz).mask         = mask_slice;
                end
        
                data_obj(kz).(fieldName) = img_slice;
        
            else
                if nargin < 4
                    data_obj(kz).isEmptySlice   = true;
                    data_obj(kz).mask           = mask_slice;
                end
                data_obj(kz).(fieldName)    = [];
            end
        
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
        
        % rescale the network parameters between the defined lower/upper bounds
        function parameters = rescale_parameters(parameters,fitParams,ub,lb)
            parameters.s0mw     = parameters.s0mw     * (ub(1)-lb(1)) + lb(1);
            parameters.s0iw     = parameters.s0iw     * (ub(2)-lb(2)) + lb(2);
            parameters.s0ew     = parameters.s0ew     * (ub(3)-lb(3)) + lb(3);
            parameters.r2smw    = parameters.r2smw    * (ub(4)-lb(4)) + lb(4);
            parameters.r2siw    = parameters.r2siw    * (ub(5)-lb(5)) + lb(5);
            parameters.r2sew    = parameters.r2sew    * (ub(6)-lb(6)) + lb(6);
            parameters.r1iew    = parameters.r1iew    * (ub(7)-lb(7)) + lb(7);
            parameters.kiewm    = parameters.kiewm    * (ub(8)-lb(8)) + lb(8);
            parameters.freq_mw  = parameters.freq_mw  * (ub(9)-lb(9)) + lb(9);
            parameters.freq_iw  = parameters.freq_iw  * (ub(10)-lb(10)) + lb(10);

            if fitParams.isComplex
                parameters.totalField   = parameters.totalField  * (ub(11)-lb(11)) + lb(11);
                parameters.pini         = parameters.pini  * (ub(12)-lb(12)) + lb(12);
            end

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

        function [temp_dir,temp_prefix, identifier] = setup_temp_dir(imgPara,temp_prefix)
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
                output_dir = fullfile(pwd,'mwi_results');
            end
            if ~exist(output_dir,'dir')
                mkdir(output_dir)
            end
            
            % create a new identifier if not provided
            identifier = [];
            % make sure the identifier is unique
            while isempty(identifier) || exist([temp_prefix identifier '.mat'],'file')
                identifier = datestr(datetime('now'),'yymmddHHMMSSFFF');
            end
            
            temp_dir        =  fullfile(output_dir,['temporary_' identifier]);
            temp_prefix   = [temp_prefix identifier];
            
            if ~exist(temp_dir,'dir')
                mkdir(temp_dir)
            end
            
        end
    
    end

end