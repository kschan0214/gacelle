classdef gpuNEXI

    properties (GetAccess = public, SetAccess = protected)
        b;
        Delta;
        Nav;
    end
    
    methods

        function this = gpuNEXI(b, Delta, varargin)
%NEXIDOT Exchange rate estimation using NEXI model with dot compartment
% smt = NEXIdot(b, Delta[, Nav])
%       output:
%           - smt: object of a fitting class
%
%       input:
%           - b: b-value [ms/um2]
%           - Delta: gradient seperation [ms]
%           - Nav (optional): # gradient direction for each b-shell
%
%       usage:
%           smt = NEXIdot(b, Delta, Nav);
%           [fa, Da, De, r, fdot] = smt.fit(S);
%           Sfit = smt.FWD([fa, Da, De, r, fdot]);
%
%           smt = NEXIdot(b, Delta, Nav);
%           [x_train, S_train] = smt.traindata(1e4);
%           pars0 = smt.likelihood(S, x_train, S_train);
%           [fa, Da, De, r, fdot] = smt.fit(S, pars0);
%           Sfit = smt.FWD([fa, Da, De, r, fdot]);
%
% Author:
%  Kwok-Shing Chan (kchan2@mgh.harvard.edu) 
%  Hong-Hsi Lee (hlee84@mgh.harvard.edu)
%  Copyright (c) 2023 Massachusetts General Hospital
%
%  Adapted from the code of
%  Dmitry Novikov (dmitry.novikov@nyulangone.org)
%  Copyright (c) 2023 New York University
            
            this.b = b;
            this.Delta = Delta;
            if nargin > 2
                this.Nav = varargin{1};
            else
                this.Nav = ones(size(b));
            end
        end

        function [fa, Da, De, ra, out] = fit(this,y,mask,fitting,pars0)
            
            % check GPU
            gpuDevice;
            
            % check image size
            dims = size(y);

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
            
            % get fitting algorithm setting
            if isfield(fitting,'Nepoch')
                numEpochs = fitting.Nepoch;
            else
                numEpochs = 4000;
            end
            if isfield(fitting,'initialLearnRate')
                initialLearnRate = fitting.initialLearnRate;
            else
                initialLearnRate = 0.1;
            end
            if isfield(fitting,'decayRate')
                decayRate = fitting.decayRate;
            else
                decayRate = 0.005;
            end
            if isfield(fitting,'stepTol')
                stepTol = fitting.stepTol;
            else
                stepTol = 1e-10;
            end
            if isfield(fitting,'tol')
                tol = fitting.tol;
            else
                tol = 1e-6;
            end
            if isfield(fitting,'lambda')
                lambda = fitting.lambda;
            else
                lambda = 0;
            end
            if isfield(fitting,'TVmode')
                TVmode = fitting.TVmode;
            else
                TVmode = '2D';
            end
            if isfield(fitting,'voxelSize')
                voxelSize = fitting.voxelSize;
            else
                voxelSize = [2,2,2];
            end
            if isfield(fitting,'display')
                isdisplay = fitting.display;
            else
                isdisplay = 0;
            end
            if isfield(fitting,'randomness')
                randomness = fitting.randomness;
            else
                randomness = 0;
            end
            
            % put data input gpuArray
            mask = gpuArray(logical(mask));  
            y    = gpuArray(single(y));

            ub              = [1,   3, 3, 1];
            lb              = [eps,eps,eps,1/250];

            if nargin < 5
                % no initial starting points
                pars0 = [];
            else
                pars0 = single(pars0);
            end
            parameters      = this.initialise_model(dims(2:end),pars0,ub,lb,randomness);   % all parameters are betwwen [0,1]
         
            % display optimisation algorithm parameters
            disp(['Maximum no. of iteration = ' num2str(numEpochs)]);
            disp(['Loss tolerance           = ' num2str(tol)]);
            disp(['Loss step tolerance      = ' num2str(stepTol)]);
            disp(['Initial learning rate    = ' num2str(initialLearnRate)]);
            disp(['Learning rate decay rate = ' num2str(decayRate)]);
            if lambda > 0 
                disp(['Regularisation parameter = ' num2str(lambda)]);
            end
            
            % clear cache before running everthing
            accfun = dlaccelerate(@this.modelGradients);
            clearCache(accfun)

            % optimisation process
            averageGrad     = [];
            averageSqGrad   = [];
            
            if isdisplay
                figure
                C = colororder;
                lineLoss = animatedline('Color',C(2,:));
                ylim([0 inf])
                xlabel("Iteration")
                ylabel("Loss")
                grid on
                
            end
            start = tic;

            minLoss         = inf;
            loss_last       = 0;
            loss            = inf;
            loss_step       = 1+stepTol;
            % optimisation
            for epoch = 1:numEpochs

                % Lower bound                               % upper bound
                parameters.fa   = max(parameters.fa,0);     parameters.fa   = min(parameters.fa,1)  ;
                parameters.Da   = max(parameters.Da,0);     parameters.Da   = min(parameters.Da,1)  ;
                parameters.De   = max(parameters.De,0);     parameters.De   = min(parameters.De,1)  ;
                parameters.ra   = max(parameters.ra,0);     parameters.ra   = min(parameters.ra,1)  ;
                
                % Evaluate the model gradients and loss using dlfeval and the modelGradients function.
                [gradients,loss] = dlfeval(accfun,parameters,y,mask,ub,lb,numMaskVox,lambda,TVmode,voxelSize);
            
                % Update learning rate.
                learningRate = initialLearnRate / (1+decayRate*epoch);
                
                % get loss
                loss_step   = abs(loss_last - loss);
                loss_last   = loss;
                if minLoss > loss
                    minLoss = loss;
                    parameters_minLoss = parameters;
                end
                % check if the optimisation should be stopped
                if loss_step < stepTol 
                    fprintf('Loss step size is less than the tolerance %e \n',stepTol);
                    break
                end
                if loss < tol
                    fprintf('Loss is less than the tolerance %e \n',tol);
                    break
                end

                % Update the network parameters using the adamupdate function.
                [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
                    averageSqGrad,epoch,learningRate);
                
                loss        = double(gather(extractdata(loss)));
                loss_step   = double(gather(extractdata(loss_step)));
                if isdisplay
                    
                    addpoints(lineLoss,epoch, loss);
                
                    D = duration(0,0,toc(start),'Format','hh:mm:ss');
                    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
                    drawnow
                end
                if mod(epoch,100) == 0 || epoch == 1
                    % display some info
                    D = duration(0,0,toc(start),'Format','hh:mm:ss');
                    fprintf('Iteration #%d,     Loss = %f,      Loss step = %e,     Elapsed:%s \n',epoch,loss,loss_step,string(D));
                end
                
            end
            fprintf('Final loss         =  %e\n',double(loss));
            fprintf('Final loss step    =  %e\n',double(loss_step));
            fprintf('Final #iterations  =  %d\n',epoch);
            
            % make sure the final results stay within boundary
            % Lower bound                               % upper bound
            parameters.fa   = max(parameters.fa,0)  ;   parameters.fa   = min(parameters.fa,1)  ;
            parameters.Da   = max(parameters.Da,0)  ;   parameters.Da   = min(parameters.Da,1)  ;
            parameters.De   = max(parameters.De,0)  ;   parameters.De   = min(parameters.De,1)  ;
            parameters.ra   = max(parameters.ra,0)  ;   parameters.ra   = min(parameters.ra,1)  ;
            
            % rescale the network parameters
            fa      = double(gather(extractdata(parameters.fa.* mask(1,:,:,:))))    * (ub(1)-lb(1)) + lb(1) ; fa    = reshape(fa, [dims(2:end) 1]);
            Da      = double(gather(extractdata(parameters.Da.* mask(1,:,:,:))))    * (ub(2)-lb(2)) + lb(2) ; Da    = reshape(Da, [dims(2:end) 1]);
            De      = double(gather(extractdata(parameters.De.* mask(1,:,:,:))))    * (ub(3)-lb(3)) + lb(3) ; De    = reshape(De, [dims(2:end) 1]);
            ra      = double(gather(extractdata(parameters.ra.* mask(1,:,:,:))))    * (ub(4)-lb(4)) + lb(4) ; ra    = reshape(ra, [dims(2:end) 1]);
            
            % result at final iteration
            out.final.fa = fa;
            out.final.Da = Da;
            out.final.De = De;
            out.final.ra = ra;
            out.final.loss = double(gather(extractdata(loss_last)));
            
            % result at minimum loss
            out.min.fa      = double(gather(extractdata(parameters_minLoss.fa.* mask(1,:,:,:))))    * (ub(1)-lb(1)) + lb(1);  out.min.fa    = reshape(out.min.fa, [dims(2:end) 1]);
            out.min.Da      = double(gather(extractdata(parameters_minLoss.Da.* mask(1,:,:,:))))    * (ub(2)-lb(2)) + lb(2) ; out.min.Da    = reshape(out.min.Da, [dims(2:end) 1]);
            out.min.De      = double(gather(extractdata(parameters_minLoss.De.* mask(1,:,:,:))))    * (ub(3)-lb(3)) + lb(3) ; out.min.De    = reshape(out.min.De, [dims(2:end) 1]);
            out.min.ra      = double(gather(extractdata(parameters_minLoss.ra.* mask(1,:,:,:))))    * (ub(4)-lb(4)) + lb(4) ; out.min.ra    = reshape(out.min.ra, [dims(2:end) 1]);
            out.min.loss    = double(gather(extractdata(minLoss)));
            
            disp('The processing is completed.')
            
            % clear GPU
            if gpuDeviceCount > 0
                gpuDevice([]);
            end

        end

        function [gradients,loss] = modelGradients(this, parameters, dlR, mask, ub,lb, numMaskVox, lambda, TVmode,voxelSize)

            % scaling network parameter
            parameters.fa   = parameters.fa  * (ub(1)-lb(1)) + lb(1);
            parameters.Da   = parameters.Da  * (ub(2)-lb(2)) + lb(2);
            parameters.De   = parameters.De  * (ub(3)-lb(3)) + lb(3);
            parameters.ra   = parameters.ra  * (ub(4)-lb(4)) + lb(4);
            
            % Make predictions with the initial conditions.
            R = this.FWD(parameters);
            R(isinf(R)) = 0;
            R(isnan(R)) = 0;

            % Masking
            R   = dlarray(R(mask>0).',     'CB');
            dlR = dlarray(dlR(mask>0).',   'CB');

            % Data fidelity term
            % loss_fidelity = mse(R, dlR);
            loss_fidelity = l1loss(R, dlR);
            
            % regularisation term
            if lambda > 0
                cost = this.reg_TV(squeeze(parameters.fa),squeeze(mask(1,:,:,:)),TVmode,voxelSize);
                loss_reg      = sum(abs(cost),"all")/numMaskVox *lambda;
                % lambda = 0.001;
            else
                loss_reg = 0;
            end
            
            % compute loss
            loss = loss_fidelity + loss_reg;
            
            % Calculate gradients with respect to the learnable parameters.
            gradients = dlgradient(loss,parameters);
        
        end

        function [s] = FWD(this, pars)
        % Forward model to generate NEXI signal
                fa   = pars.fa;
                Da   = pars.Da;
                De   = pars.De;
                ra   = pars.ra;
                
                % Forward model
                s = this.S(fa, Da, De, ra);
                
        end
        
        % NEXI signal
        function S = S(this, fa, Da, De, ra)
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

        function cost = reg_TV(this,img,mask,TVmode,voxelSize)
            % voxel_size = [1 1 1];
            % Vr      = 1./sqrt(abs(mask.*this.gradient_operator(img,voxel_size)).^2+eps);
            cost = sum(abs(mask.*this.gradient_operator(img,voxelSize,TVmode)),4);

            % cost    = this.divergence_operator(mask.*(Vr.*(mask.*this.gradient_operator(img,voxel_size))),voxel_size);
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

        function parameters = initialise_model(img_size,pars0,ub,lb,randomness)

            % Initialize the parameters for the first fully connect operation.
            parameters = struct;
            
            if isempty(pars0)
                % 1st dimension preserves for DWI data points 
                % initialise model parameters randomly
                parameters.fa   = gpuArray( dlarray(rand([1 img_size],'single') ));     % values between [0,1]
                parameters.Da   = gpuArray( dlarray(rand([1 img_size],'single') ));     % values between [0,1]
                parameters.De   = gpuArray( dlarray(rand([1 img_size],'single') ));     % values between [0,1]
                parameters.ra   = gpuArray( dlarray(rand([1 img_size],'single') ));     % values between [0,1]
            else
                parameters.fa   = gpuArray( dlarray( pars0(1,:,:,:) )) /(ub(1)-lb(1));     % values between [0,1]
                parameters.Da   = gpuArray( dlarray( pars0(2,:,:,:) )) /(ub(2)-lb(2));     % values between [0,1]
                parameters.De   = gpuArray( dlarray( pars0(3,:,:,:) )) /(ub(3)-lb(3));     % values between [0,1]
                parameters.ra   = gpuArray( dlarray( pars0(4,:,:,:) )) /(ub(4)-lb(4));     % values between [0,1]

                % For noise propagation add a bit randomness to avoid trapped at initial points
                % randomness = 1/3; % 1: totally random; 0: use entirely the prior
                parameters.fa   = (1-randomness)*parameters.fa + randomness*gpuArray( dlarray(rand([1 img_size],'single') ));
                parameters.Da   = (1-randomness)*parameters.Da + randomness*gpuArray( dlarray(rand([1 img_size],'single') ));
                parameters.De   = (1-randomness)*parameters.De + randomness*gpuArray( dlarray(rand([1 img_size],'single') ));
                parameters.ra   = (1-randomness)*parameters.ra + randomness*gpuArray( dlarray(rand([1 img_size],'single') ));
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