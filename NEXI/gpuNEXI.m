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

            gpuDevice;

            dims = size(y);

            if nargin == 2 || isempty(mask)
                mask = ones(dims);
            else
                mask = permute(repmat(mask,[1 1 1 dims(1)]),[4 1 2 3]);
            end
            if nargin < 4
                fitting = struct();
            end
            numMaskVox = numel(mask(mask ~= 0)) / dims(1);

            mask = gpuArray(single(mask));  
            y    = gpuArray(single(y));

            if nargin == 5
                % if prior is provided
                parameters.fa   = gpuArray( dlarray(pars0(1,:,:,:)));
                parameters.Da   = gpuArray( dlarray(pars0(2,:,:,:)));
                parameters.De   = gpuArray( dlarray(pars0(3,:,:,:)));
                parameters.ra   = gpuArray( dlarray(pars0(4,:,:,:)));
            else
                % otherwise initialise model parameters randomly
                parameters = this.initialise_model(dims(2:end));
            end
    
            % clear cache before running everthing
            accfun = dlaccelerate(@this.modelGradients);
            clearCache(accfun)
            
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
            if isfield(fitting,'display')
                isdisplay = fitting.display;
            else
                isdisplay = 0;
            end

            % display optimisation algorithm parameters
            disp(['Maximum no. of iteration = ' num2str(numEpochs)]);
            disp(['Loss tolerance           = ' num2str(tol)]);
            disp(['Loss step tolerance      = ' num2str(stepTol)]);
            disp(['Initial learning rate    = ' num2str(initialLearnRate)]);
            disp(['Learning rate decay rate = ' num2str(decayRate)]);
            if lambda > 0 
                disp(['Regularisation parameter = ' num2str(lambda)]);
            end
    
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
                
                start = tic;
            end

            scaleFactor     = [0.5, 3, 1, 0.1];
            ub              = [1,   3, 3, 1];
            lb              = [eps,eps,eps,1/250];
            
            minLoss         = inf;
            loss_last       = 0;
            loss_gradient   = 1+stepTol;
            % optimisation
            for epoch = 1:numEpochs

                if loss_gradient > stepTol || tol < loss

                    % Lower bound
                    parameters.fa   = max(parameters.fa,lb(1))  ;
                    parameters.Da   = max(parameters.Da,lb(2))  ;   
                    parameters.De   = max(parameters.De,lb(3))  ;
                    parameters.ra   = max(parameters.ra,lb(4))  ;
                    % upper bound
                    parameters.fa   = min(parameters.fa,ub(1)/scaleFactor(1))  ;
                    parameters.Da   = min(parameters.Da,ub(2)/scaleFactor(2))  ;
                    parameters.De   = min(parameters.De,ub(3)/scaleFactor(3))  ;
                    parameters.ra   = min(parameters.ra,ub(4)/scaleFactor(4))  ;
                    
                    % Evaluate the model gradients and loss using dlfeval and the modelGradients function.
                    [gradients,loss] = dlfeval(accfun,parameters,y,mask,scaleFactor,numMaskVox,lambda,TVmode);
                
                    % Update learning rate.
                    learningRate = initialLearnRate / (1+decayRate*epoch);

                    loss = loss/numMaskVox;
                
                    loss_gradient   = abs(loss_last - loss);
                    loss_last       = loss;
                    if minLoss > loss
                        minLoss = loss;
                        parameters_minLoss = parameters;
                    end

                    % Update the network parameters using the adamupdate function.
                    [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
                        averageSqGrad,epoch,learningRate);
    
                    if isdisplay
                        loss = double(gather(extractdata(loss)));
                        addpoints(lineLoss,epoch, loss);
                    
                        D = duration(0,0,toc(start),'Format','hh:mm:ss');
                        title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
                        drawnow
                    end

                end
                
            end
            
            % make sure the final results stay within boundary
            % Lower bound
            parameters.fa   = max(parameters.fa,lb(1))  ;
            parameters.Da   = max(parameters.Da,lb(2))  ;   
            parameters.De   = max(parameters.De,lb(3))  ;
            parameters.ra   = max(parameters.ra,lb(4))  ;
            % upper bound
            parameters.fa   = min(parameters.fa,ub(1)/scaleFactor(1))  ;
            parameters.Da   = min(parameters.Da,ub(2)/scaleFactor(2))  ;
            parameters.De   = min(parameters.De,ub(3)/scaleFactor(3))  ;
            parameters.ra   = min(parameters.ra,ub(4)/scaleFactor(4))  ;
            
            fa      = double(gather(extractdata(parameters.fa.* mask(1,:,:,:))))    * scaleFactor(1) ; fa    = reshape(fa, [dims(2:end) 1]);
            Da      = double(gather(extractdata(parameters.Da.* mask(1,:,:,:))))    * scaleFactor(2) ; Da    = reshape(Da, [dims(2:end) 1]);
            De      = double(gather(extractdata(parameters.De.* mask(1,:,:,:))))    * scaleFactor(3) ; De    = reshape(De, [dims(2:end) 1]);
            ra      = double(gather(extractdata(parameters.ra.* mask(1,:,:,:))))    * scaleFactor(4) ; ra    = reshape(ra, [dims(2:end) 1]);
            
            out.final.fa = fa;
            out.final.Da = Da;
            out.final.De = De;
            out.final.ra = ra;
            out.final.loss = double(gather(extractdata(loss_last)));

            out.min.fa      = double(gather(extractdata(parameters_minLoss.fa.* mask(1,:,:,:))))    * scaleFactor(1) ; out.min.fa    = reshape(out.min.fa, [dims(2:end) 1]);
            out.min.Da      = double(gather(extractdata(parameters_minLoss.Da.* mask(1,:,:,:))))    * scaleFactor(2) ; out.min.Da    = reshape(out.min.Da, [dims(2:end) 1]);
            out.min.De      = double(gather(extractdata(parameters_minLoss.De.* mask(1,:,:,:))))    * scaleFactor(3) ; out.min.De    = reshape(out.min.De, [dims(2:end) 1]);
            out.min.ra      = double(gather(extractdata(parameters_minLoss.ra.* mask(1,:,:,:))))    * scaleFactor(4) ; out.min.ra    = reshape(out.min.ra, [dims(2:end) 1]);
            out.min.loss    = double(gather(extractdata(minLoss)));
            
            disp('The processing is completed.')
            
            if gpuDeviceCount > 0
                gpuDevice([]);
            end

        end

        function [gradients,loss] = modelGradients(this, parameters, dlR, mask, scaleFactor, numMaskVox, lambda, TVmode)

            % scaling network parameter
            parameters.fa   = parameters.fa  * scaleFactor(1);
            parameters.Da   = parameters.Da  * scaleFactor(2);
            parameters.De   = parameters.De  * scaleFactor(3);
            parameters.ra   = parameters.ra  * scaleFactor(4);
            
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
                cost = this.reg_TV(squeeze(parameters.Da),squeeze(mask(1,:,:,:)),TVmode);
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

        function cost = reg_TV(this,img,mask,TVmode)
            voxel_size = [1 1 1];
            % Vr      = 1./sqrt(abs(mask.*this.gradient_operator(img,voxel_size)).^2+eps);
            cost = sum(abs(mask.*this.gradient_operator(img,voxel_size,TVmode)),4);

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

        function parameters = initialise_model(img_size)

            % Initialize the parameters for the first fully connect operation.
            parameters = struct;

            parameters.fa   = gpuArray( dlarray(rand([1 img_size],'single') ));
            parameters.Da   = gpuArray( dlarray(rand([1 img_size],'single') )) ;
            parameters.De   = gpuArray( dlarray(rand([1 img_size],'single') )) ;
            parameters.ra   = gpuArray( dlarray(rand([1 img_size],'single') ));
            
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