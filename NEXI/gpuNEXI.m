classdef gpuNEXI
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 8 Dec 2023 (v1.0.0)
% Date modified:

    properties (GetAccess = public, SetAccess = protected)
        b;
        Delta;
        Nav;
    end
    
    methods

        % constructuor
        function this = gpuNEXI(b, Delta, varargin)
        % NEXI Exchange rate estimation using NEXI model
        % obj = gpuNEXI(b, Delta, Nav)
        %
        % Input
        % ----------
        % b         : b-value [ms/um2]
        % Delta     : gradient seperation [ms]
        % Nav       : # gradient direction for each b-shell (optional)
        %
        % Output
        % ----------
        % obj       : object of a fitting class
        %
        % Usage
        % ----------
        % obj                   = NEXI(b, Delta, Nav);
        % [out, fa, Da, De, r]  = obj.fit(S, mask, fitting,);
        % Sfit                  = smt.FWD([fa, Da, De, r]);
        % [x_train, S_train]    = obj.traindata(1e4);
        % pars0                 = smt.likelihood(S, x_train, S_train);
        % [out, fa, Da, De, r]  = smt.fit(S, mask, fitting, pars0);
        %
        % Author:
        %  Kwok-Shing Chan (kchan2@mgh.harvard.edu) 
        %  Hong-Hsi Lee (hlee84@mgh.harvard.edu)
        %  Copyright (c) 2023 Massachusetts General Hospital
        %
        %  Adapted from the code of
        %  Dmitry Novikov (dmitry.novikov@nyulangone.org)
        %  Copyright (c) 2023 New York University
            
            this.b = b(:);
            this.Delta = Delta(:);
            if nargin > 2
                this.Nav = varargin{1};
            else
                this.Nav = ones(size(b));
            end
            this.Nav = this.Nav(:);
        end
    
        % automatically segment data and fitting in case the data cannot fit in the GPU in one go
        function  [out, fa, Da, De, ra, p2] = estimate(this, dwi, mask, fitting, pars0)
            gpuDevice([]);
            gpu = gpuDevice;

            dims = size(dwi);

            maxMemory = floor(gpu.TotalMemory / 1024^3)*1024^3;


            memoryRequiredFixPerVoxel       = 40 * prod(dims);
            memoryRequiredDynamicperVoxel   = 1.5e3 * prod(dims([1,2,4]));

            maxSlice = floor((maxMemory - memoryRequiredFixPerVoxel)/memoryRequiredDynamicperVoxel);
            NSegment = ceil(dims(3)/maxSlice);

            fprintf('Data is divided into %d segments\n',NSegment);
            
            fa = zeros(dims(1:3));
            Da = zeros(dims(1:3));
            De = zeros(dims(1:3));
            ra = zeros(dims(1:3));
            p2 = zeros(dims(1:3));
            for ks = 1:NSegment

                fprintf('Running #Segment = %d/%d \n',ks,NSegment);
                disp   ('------------------------')
    
                slice = 1+(ks-1)*dims(3)/NSegment : ks*dims(3)/NSegment;
                
                dwi_tmp     = dwi(:,:,slice,:);
                mask_tmp    = mask(:,:,slice);
                pars0_tmp   = pars0(:,:,slice,:);
                if fitting.lmax >0
                    [out(ks), fa(:,:,slice), Da(:,:,slice), De(:,:,slice), ra(:,:,slice), p2(:,:,slice)]  = this.fit(dwi_tmp,mask_tmp,fitting,pars0_tmp);
                else
                    [out(ks), fa(:,:,slice), Da(:,:,slice), De(:,:,slice), ra(:,:,slice)]  = this.fit(dwi_tmp,mask_tmp,fitting,pars0_tmp);
                    p2 = [];
                end
            end

        end

        % Data fitting function
        function [out, fa, Da, De, ra, p2] = fit(this,dwi,mask,fitting,pars0)
        %
        % Input
        % -----------
        % dwi       : S0 normalised 4D dwi images, [x,y,slice,diffusion], 4th dimension corresponding to [Sl0_b1,Sl0_b2,Sl2_b1,Sl2_b2, etc.]; the order of bval must match the order in the constructor gpuNEXI
        % mask      : 3D signal mask, [x,y,slice]
        % fitting   : fitting algorithm parameters
        %   .Nepoch             : no. of maximum iterations, default = 4000
        %   .initialLearnRate   : initial gradient step size, defaulr = 0.01
        %   .decayRate          : decay rate of gradient step size; learningRate = initialLearnRate / (1+decayRate*epoch), default = 0.0005
        %   .convergenceValue   : convergence tolerance, based on the slope of last 'convergenceWindow' data points on loss, default = 1e-8
        %   .convergenceWindow  : number of data points to check convergence, default = 20
        %   .tol                : stop criteria on metric value, default = 1e-3
        %   .lambda             : regularisation parameter, default = 0 (no regularisation)
        %   .TVmode             : mode for TV regulariation, '2D'|'3D', default = '2D'
        %   .regmap             : parameter map used for regularisation, 'fa'|'ra'|'Da'|'De', default = 'fa'
        %   .lmax               : Order of rotational invariant, 0|2, default = 0
        %   .lossFunction       : loss for data fidelity term, 'L1'|'L2'|'MSE', default = 'L1'
        %   .display            : online display the fitting process on figure, true|false, defualt = false
        % pars0     : 4D parameter starting points of fitting, [x,y,slice,param], 4th dimension corresponding to fitting  parameters with order [fa,Da,De,ra,p2] (optional)
        % 
        % Output
        % -----------
        % out       : output structure
        %   .final      : final results
        %       .fa         : Intraneurite volume fraction
        %       .Da         : Intraneurite diffusivity (um2/ms)
        %       .De         : Extraneurite diffusivity (um2/ms)
        %       .ra         : exchange rate from intra- to extra-neurite compartment
        %       .p2         : dispersion index (if fitting.lax=2)
        %       .loss       : final loss metric
        %   .min        : results with the minimum loss metric across all iterations
        %       .fa         : Intraneurite volume fraction
        %       .Da         : Intraneurite diffusivity (um2/ms)
        %       .De         : Extraneurite diffusivity (um2/ms)
        %       .ra         : exchange rate from intra- to extra-neurite compartment
        %       .p2         : dispersion index (if fitting.lax=2)
        %       .loss       : loss metric      
        % fa        : final Intraneurite volume fraction
        % Da        : final Intraneurite diffusivity (um2/ms)
        % De        : final Extraneurite diffusivity (um2/ms)
        % ra        : final exchange rate from intra- to extra-neurite compartment
        % p2        : final dispersion index (if fitting.lax=2)
        %
        % Description: askAdam Image-based NEXI model fitting
        %
        % Kwok-Shing Chan @ MGH
        % kchan2@mgh.harvard.edu
        % Date created: 8 Dec 2023
        % Date modified:
        %
        %
            
            % check GPU
            gpuDevice;
            
            % check image size
            dwi     = permute(dwi,[4 1 2 3]);
            dims    = size(dwi);

            if nargin < 3 || isempty(mask)
                % if no mask input then fit everthing
                mask = ones(dims);
            else
                % assume mask is 3D
                mask = permute(repmat(mask,[1 1 1 dims(1)]),[4 1 2 3]);
            end
            numMaskVox = numel(mask(mask ~= 0)) / dims(1);

            if nargin < 4
                fitting = struct();
            end

            % get all fitting algorithm parameters 
            fitting = this.check_set_default(fitting);

            % put data input gpuArray
            mask = gpuArray(logical(mask));  
            dwi  = gpuArray(single(dwi));
            
            % set fitting boundary
            ub  = [1,   3, 3, 1];
            lb  = [eps,eps,eps,1/250];
            if fitting.lmax == 2
                ub = cat(2,ub, 1);
                lb = cat(2,lb, eps);
            end
            
            % set initial tarting points
            if nargin < 5
                % no initial starting points
                pars0 = [];
            else
                pars0 = permute(pars0,[4 1 2 3]);
                pars0 = single(pars0);
            end
            parameters = this.initialise_model(dims(2:end),pars0,ub,lb,fitting);   % all parameters are betwwen [0,1]
         
            l = 0:2:fitting.lmax;
            w = zeros(size(dwi));
            for kl = 1:(fitting.lmax/2+1)
                for kb = 1:numel(this.b)
                    w((kl-1)*numel(this.b)+kb,:,:,:) = this.Nav(kb) / (2*l(kl)+1);
                end
            end
            if strcmpi(fitting.lossFunction,'l1')
                w = sqrt(w);
            end
            w = w ./ max(w(:));
            w = w(mask>0);
            w = dlarray(gpuArray(w).','CB');

            % display optimisation algorithm parameters
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
            disp(['lmax                     = ' num2str(fitting.lmax)]);
            
            % clear cache before running everthing
            accfun = dlaccelerate(@this.modelGradients);
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
                [gradients,loss,loss_fidelity,loss_reg] = dlfeval(accfun,parameters,dwi,mask,w,ub,lb,numMaskVox,fitting);
            
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
            parameters  = this.rescale_parameters(parameters,lb,ub);
            fa          = single(gather(extractdata(parameters.fa.* mask(1,:,:,:)))); fa    = reshape(fa, [dims(2:end) 1]);
            Da          = single(gather(extractdata(parameters.Da.* mask(1,:,:,:)))); Da    = reshape(Da, [dims(2:end) 1]);
            De          = single(gather(extractdata(parameters.De.* mask(1,:,:,:)))); De    = reshape(De, [dims(2:end) 1]);
            ra          = single(gather(extractdata(parameters.ra.* mask(1,:,:,:)))); ra    = reshape(ra, [dims(2:end) 1]);
            
            % result at final iteration
            out.final.fa = fa;
            out.final.Da = Da;
            out.final.De = De;
            out.final.ra = ra;
            out.final.loss          = loss;
            out.final.loss_fidelity = double(gather(extractdata(loss_fidelity)));
            if fitting.lambda == 0
                out.final.loss_reg      = 0;
            else
                out.final.loss_reg      = double(gather(extractdata(loss_reg)));
            end
            
            % result at minimum loss
            parameters_minLoss      = this.rescale_parameters(parameters_minLoss,lb,ub);
            out.min.fa              = single(gather(extractdata(parameters_minLoss.fa.* mask(1,:,:,:))));   out.min.fa    = reshape(out.min.fa, [dims(2:end) 1]);
            out.min.Da              = single(gather(extractdata(parameters_minLoss.Da.* mask(1,:,:,:))));   out.min.Da    = reshape(out.min.Da, [dims(2:end) 1]);
            out.min.De              = single(gather(extractdata(parameters_minLoss.De.* mask(1,:,:,:))));   out.min.De    = reshape(out.min.De, [dims(2:end) 1]);
            out.min.ra              = single(gather(extractdata(parameters_minLoss.ra.* mask(1,:,:,:))));   out.min.ra    = reshape(out.min.ra, [dims(2:end) 1]);
            out.min.loss            = minLoss;
            out.min.loss_fidelity   = double(gather(extractdata(minLossFidelity)));
            if fitting.lambda == 0
                out.min.loss_reg      = 0;
            else
                out.min.loss_reg      = double(gather(extractdata(minLossRegularisation)));
            end

            if fitting.lmax == 2
                p2              = single(gather(extractdata(parameters.p2.* mask(1,:,:,:)))); p2    = reshape(p2, [dims(2:end) 1]);
                out.final.p2    = p2;
                out.min.p2      = single(gather(extractdata(parameters_minLoss.p2.* mask(1,:,:,:)))); out.min.p2    = reshape(out.min.p2, [dims(2:end) 1]);
            else
                p2 = [];
            end
            
            disp('The processing is completed.')
            
            % clear GPU
            if gpuDeviceCount > 0
                gpuDevice([]);
            end

        end

        % compute the gradient and loss of forward modelling
        function [gradients,loss,loss_fidelity,loss_reg] = modelGradients(this, parameters, dlR, mask, weights, ub,lb, numMaskVox, fitting)

            % rescale network parameter to true values
            parameters = this.rescale_parameters(parameters,lb,ub);
            
            % Forward model
            R           = this.FWD(parameters,fitting);
            R(isinf(R)) = 0;
            R(isnan(R)) = 0;

            % Masking
            R   = dlarray(R(mask>0).',     'CB');
            dlR = dlarray(dlR(mask>0).',   'CB');

            % Data fidelity term
            switch lower(fitting.lossFunction)
                case 'l1'
                    loss_fidelity = l1loss(R, dlR, weights);
                case 'l2'
                    loss_fidelity = l2loss(R, dlR, weights);
                case 'mse'
                    loss_fidelity = mse(R, dlR);
            end
            
            % regularisation term
            if fitting.lambda > 0
                cost        = this.reg_TV(squeeze(parameters.(fitting.regmap)),squeeze(mask(1,:,:,:)),fitting.TVmode,fitting.voxelSize);
                loss_reg    = sum(abs(cost),"all")/numMaskVox *fitting.lambda;
            else
                loss_reg = 0;
            end
            
            % compute loss
            loss = loss_fidelity + loss_reg;
            
            % Calculate gradients with respect to the learnable parameters.
            gradients = dlgradient(loss,parameters);
        
        end
        
        % NEXI signal
        % copmpute the forward model
        function [s] = FWD(this, pars, fitting)
        % Forward model to generate NEXI signal
                fa   = pars.fa;
                Da   = pars.Da;
                De   = pars.De;
                ra   = pars.ra;
                
                % Forward model
                % Sl0
                s = this.Sl0(fa, Da, De, ra);
                
                % Sl2
                if fitting.lmax == 2
                    p2 = pars.p2;
                    s = cat(1,s,this.Sl2(fa, Da, De, ra, p2));
                end

                % make sure s cannot be greater than 1
                s = min(s,1);
                
        end
        
        % 0th order rotational invariant
        function S = Sl0(this, fa, Da, De, ra)
            Da = this.b.*Da;
            De = this.b.*De;
            ra = this.Delta.*ra;
            re = ra.*fa./(1-fa);
            
            % Trapezoidal's rule replacement
            Nx  = 14;    % NRMSE<0.05% for Nx=14
            x   = zeros([ones(1,ndims(fa)), Nx]); x(:) = linspace(0,1,Nx);
            S   = trapz(x(:),this.M(x, fa, Da, De, ra, re),ndims(x));
            % myfun = @(x) this.M(x, fa, Da, De, ra, re);
            % S = integral(myfun, 0, 1, 'AbsTol', 1e-14, 'ArrayValued', true);

        end
        % 2nd order rotational invariant
        function S = Sl2(this, fa, Da, De, ra, p2)
            Da = this.b.*Da;
            De = this.b.*De;
            ra = this.Delta.*ra;
            re = ra.*fa./(1-fa);
            
            % Trapezoidal's rule replacement
            Nx  = 14;    % NRMSE<0.5% for Nx=14
            x   = zeros([ones(1,ndims(fa)), Nx]); x(:) = linspace(0,1,Nx);
            S   = trapz(x(:),this.M(x, fa, Da, De, ra, re).*(3*x.^2-1)/2,ndims(x));
            S   = p2.*abs(S);

        end

        % Total variation regularisation
        function cost = reg_TV(this,img,mask,TVmode,voxelSize)
            % voxel_size = [1 1 1];
            % Vr      = 1./sqrt(abs(mask.*this.gradient_operator(img,voxel_size)).^2+eps);
            cost = sum(abs(mask.*this.gradient_operator(img,voxelSize,TVmode)),4);

            % cost    = this.divergence_operator(mask.*(Vr.*(mask.*this.gradient_operator(img,voxel_size))),voxel_size);
        end
        
        % likelihood
        function [x_train, S_train, intervals] = traindata(this, N_samples, lmax, varargin)
            if nargin < 4
                intervals = [0.01 0.99  ;   % fa
                              1.5 3     ;   % Da
                              0.5 1.5   ;   % De
                                1 100   ;   % exchange time = (1-fa)/r
                             0.01 0.99 ];   % p2
            else
                intervals = varargin{1};
            end
            
            if lmax == 0
                numBSample = numel(this.b);
                numParam   = size(intervals,1) - 1;
            elseif lmax == 2
                numBSample = numel(this.b)*2;
                numParam   = size(intervals,1);
            end
            
            % batch size can be modified according to available hardware
            batch_size  = 1e3;
            reps        = ceil(N_samples/batch_size);
            x_train     = zeros(numParam,batch_size,reps);
            S_train     = zeros(numBSample,batch_size,reps);
            for k = 1:reps
                % generate random parameter guesses and construct batch for NN signal evaluation
                pars = intervals(:,1) + diff(intervals,[],2).*rand(size(intervals,1),batch_size);
                % pars(3,:) = pars(2,:).*pars(3,:);
                pars(4,:) = 1./pars(4,:).*(1-pars(1,:));

                % NEXI Krger signal evaluation
                Sl0 = zeros(numel(this.b),batch_size);
                for j = 1:batch_size
                    Sl0(:,j) = this.Sl0(pars(1,j), pars(2,j), pars(3,j), pars(4,j));
                end

                % in case of Sl2
                if lmax == 2
                    Sl2 = zeros(numel(this.b),batch_size);
                    for j = 1:batch_size
                        Sl2(:,j) = this.Sl2(pars(1,j), pars(2,j), pars(3,j), pars(4,j), pars(5,j)) ;
                    end

                else
                    pars(5,:)   = [];
                    Sl2         = [];
                end

                % remaining signals (dot, soma)
                x_train(:,:,k) = pars;
                S_train(:,:,k) = cat(1,Sl0,Sl2);

            end
            % intervals(3,:) = intervals(2,:).*intervals(3,:);
            intervals(4,:) = (1-intervals(1,end:-1:1))./intervals(4,end:-1:1);
            if lmax == 2
                intervals(5,:) = [];
            end
        end
    
        function [pars_best, sse_best] = likelihood(this, S0, x_train, S_train,lmax)
            wt = kron(this.Nav(:), 1./(2*(0:2:lmax)+1));
            wt = wt(:);
            nL = floor(lmax/2);
            S0 = S0(1:numel(this.b)*(nL+1),:);
            % batch size can be modified according to available hardware
            [Nx, ~, reps] = size(x_train);
            [~, Nv] = size(S0);
            pars_best = zeros(Nx,Nv);
            sse_best  = inf(1, Nv);
            for k = 1:reps
                pars = x_train(:,:,k);
                S    = S_train(:,:,k);
                for i = 1:Nv
                    S0i = S0(:,i);

                    % scale generated signals (fit S0) to input signal
                    sse = sum(wt.*(S0i - (S0i'*S)./dot(S,S).*S).^2);

                    % store best encountered parameter combination
                    [sse_new,best_index] = min(sse);
                    if sse_new<sse_best(i)
                        sse_best(i)    = sse_new;
                        pars_best(:,i) = pars(:,best_index);
                    end
                end
            end
        end

    end

    methods(Static)
        function M = M(x, fa, Da, d2, r1, r2)
            d1 = Da.*x.^2;
            l1 = (r1+r2+d1+d2)/2;
            l2 = sqrt( (r1-r2+d1-d2).^2 + 4*r1.*r2 )/2; 
            lm = l1-l2;
            Pp = (fa.*d1 + (1-fa).*d2 - lm)./(l2*2);
            M  = Pp.*exp(-(l1+l2)) + (1-Pp).*exp(-lm); 
        end

        % initialise network parameters
        function parameters = initialise_model(img_size,pars0,ub,lb,fitting)
            
            % get relevant parameters
            randomness = fitting.randomness;
            lmax       = fitting.lmax;

            % initialise model parameters randomly
            % 1st dimension preserves for DWI data points 
            fa0 = rand([1 img_size],'single') ;     % values between [0,1]
            Da0 = rand([1 img_size],'single') ;     % values between [0,1]
            De0 = rand([1 img_size],'single') ;     % values between [0,1]
            ra0 = rand([1 img_size],'single') ;     % values between [0,1]
            
            % if initial points are provided
            if ~isempty(pars0)
                % For noise propagation add a bit randomness to avoid trapped at initial points
                % randomness = 1/3; % 1: totally random; 0: use entirely the prior
                fa0 =  (1-randomness)*(pars0(1,:,:,:) /(ub(1)-lb(1))) + randomness*fa0;     % values between [0,1]
                Da0 =  (1-randomness)*(pars0(2,:,:,:) /(ub(2)-lb(2))) + randomness*Da0;     % values between [0,1]
                De0 =  (1-randomness)*(pars0(3,:,:,:) /(ub(3)-lb(3))) + randomness*De0;     % values between [0,1]
                ra0 =  (1-randomness)*(pars0(4,:,:,:) /(ub(4)-lb(4))) + randomness*ra0;     % values between [0,1]

            end
            parameters.fa = gpuArray( dlarray(fa0));
            parameters.Da = gpuArray( dlarray(Da0));
            parameters.De = gpuArray( dlarray(De0));
            parameters.ra = gpuArray( dlarray(ra0));
            
            % in case fitting for lmax=2
            if lmax == 2
                p20 = rand([1 img_size],'single') ;     % values between [0,1]
                if ~isempty(pars0)
                    p20 =  (1-randomness)*(pars0(5,:,:,:) /(ub(5)-lb(5))) + randomness*p20;     % values between [0,1]
                end
                parameters.p2 = gpuArray( dlarray(p20));
            end

        end
        
        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default(fitting)
            fitting2 = fitting;

            % get fitting algorithm setting
            if ~isfield(fitting,'Nepoch')
                fitting2.numEpochs = 4000;
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
            if ~-isfield(fitting,'regmap')
                fitting2.regmap = 'fa';
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
            if ~isfield(fitting,'lmax')
                fitting2.lmax = 0;
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
        function parameters = rescale_parameters(parameters,lb,ub)
            parameters.fa   = parameters.fa  * (ub(1)-lb(1)) + lb(1);
            parameters.Da   = parameters.Da  * (ub(2)-lb(2)) + lb(2);
            parameters.De   = parameters.De  * (ub(3)-lb(3)) + lb(3);
            parameters.ra   = parameters.ra  * (ub(4)-lb(4)) + lb(4);
            if numel(lb) == 5
                parameters.p2   = parameters.p2  * (ub(5)-lb(5)) + lb(5);
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
    end

end