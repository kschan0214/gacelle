classdef gpuNEXIdot

    properties (GetAccess = public, SetAccess = protected)
        b;
        Delta;
        Nav;
    end
    
    methods

        function this = gpuNEXIdot(b, Delta, varargin)
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

        function [fa, Da, De, ra, fdot] = fit(this,y,mask,fitting,pars0)

            dims = size(y);

            if nargin == 2 || isempty(mask)
                mask = ones(dims);
            else
                mask = permute(repmat(mask,[1 1 1 dims(1)]),[4 1 2 3]);
            end
            if nargin < 4
                fitting = struct();
            end

            if nargin == 5
                % if prior is provided
                parameters.fa   = gpuArray( dlarray(pars0(1,:,:,:)));
                parameters.Da   = gpuArray( dlarray(pars0(2,:,:,:)));
                parameters.De   = gpuArray( dlarray(pars0(3,:,:,:)));
                parameters.ra   = gpuArray( dlarray(pars0(4,:,:,:)));
                parameters.fdot = gpuArray( dlarray(pars0(5,:,:,:)));
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

            % display optimisation algorithm parameters
            disp(['Maximum no. of iteration = ' num2str(numEpochs)]);
            disp(['Loss tolerance           = ' num2str(tol)]);
            disp(['Loss step tolerance      = ' num2str(stepTol)]);
            disp(['Initial learning rate    = ' num2str(initialLearnRate)]);
            disp(['Learning rate decay rate = ' num2str(decayRate)]);
    
            % optimisation process
            averageGrad     = [];
            averageSqGrad   = [];
            
            isdisplay = 1;
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

            scaleFactor     = [0.5, 3, 1, 0.1, 0.1];
            
            loss_last       = 0;
            gradient_loss   = 1+stepTol;
            % optimisation
            for epoch = 1:numEpochs

                if gradient_loss > stepTol || tol < loss

                    % Lower bound
                    parameters.fa   = max(parameters.fa,eps)  ;
                    parameters.Da   = max(parameters.Da,eps)  ;   
                    parameters.De   = max(parameters.De,eps)  ;
                    parameters.ra   = max(parameters.ra,eps)  ;
                    parameters.fdot = max(parameters.fdot,eps);
                    % upper bound
                    parameters.fa   = min(parameters.fa,1/scaleFactor(1))  ;
                    parameters.Da   = min(parameters.Da,3/scaleFactor(2))  ;
                    parameters.De   = min(parameters.De,3/scaleFactor(3))  ;
                    parameters.ra   = min(parameters.ra,1/scaleFactor(4))  ;
                    parameters.fdot = min(parameters.fdot,1/scaleFactor(5));
                    
                    % Evaluate the model gradients and loss using dlfeval and the modelGradients function.
                    [gradients,loss] = dlfeval(accfun,parameters,y,mask,scaleFactor);
                
                    % Update learning rate.
                    learningRate = initialLearnRate / (1+decayRate*epoch);
                
                    % Update the network parameters using the adamupdate function.
                    [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
                        averageSqGrad,epoch,learningRate);

                    gradient_loss = abs(loss_last - loss);
                    loss_last = loss;
    
                    if isdisplay
                        loss = double(gather(extractdata(loss)));
                        addpoints(lineLoss,epoch, loss);
                    
                        D = duration(0,0,toc(start),'Format','hh:mm:ss');
                        title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
                        drawnow
                    end

                end
                
            end

            fa      = double(gather(extractdata(parameters.fa)))    * scaleFactor(1); fa    = reshape(fa, [dims(2:end) 1]);
            Da      = double(gather(extractdata(parameters.Da)))    * scaleFactor(2); Da    = reshape(Da, [dims(2:end) 1]);
            De      = double(gather(extractdata(parameters.De)))    * scaleFactor(3); De    = reshape(De, [dims(2:end) 1]);
            ra      = double(gather(extractdata(parameters.ra)))    * scaleFactor(4); ra    = reshape(ra, [dims(2:end) 1]);
            fdot    = double(gather(extractdata(parameters.fdot)))  * scaleFactor(5); fdot  = reshape(fdot, [dims(2:end) 1]);

        end

        function [gradients,loss] = modelGradients(this, parameters, dlR, mask, scaleFactor)

            % scaling network parameter
            parameters.fa   = parameters.fa  * scaleFactor(1);
            parameters.Da   = parameters.Da  * scaleFactor(2);
            parameters.De   = parameters.De  * scaleFactor(3);
            parameters.ra   = parameters.ra  * scaleFactor(4);
            parameters.fdot = parameters.fdot* scaleFactor(5);
            
            % Make predictions with the initial conditions.
            R = this.FWD(parameters);
            R(isinf(R)) = 0;
            R(isnan(R)) = 0;

%             R   = dlarray(R(:).',     'CB');
%             dlR = dlarray(dlR(:).',   'CB');

            R   = dlarray(R(mask>0).',     'CB');
            dlR = dlarray(dlR(mask>0).',   'CB');
            
            % compute loss
%             loss = mse(R, dlR);
            loss = l1loss(R, dlR);
            
            % Calculate gradients with respect to the learnable parameters.
            gradients = dlgradient(loss,parameters);
        
        end

        function [s] = FWD(this, pars)
        % Forward model to generate NEXI signal
                fa   = pars.fa;
                Da   = pars.Da;
                De   = pars.De;
                ra   = pars.ra;
                fdot = pars.fdot;
                
                % Forward model
                S = this.S(fa, Da, De, ra);
                
                % Combined signal
                s = (1-fdot).*S + fdot;
                           
        end

        function S = S(this, fa, Da, De, ra)
            Da = this.b.*Da;
            De = this.b.*De;
            ra = this.Delta.*ra;
            re = ra.*fa./(1-fa);
            
            % Trapezoidal's rule replacement
            Nx = 14;    % NRMSE<0.05% for Nx=14
            x       = zeros([ones(1,ndims(fa)), Nx]);
            x(:)    = linspace(0,1,Nx);
            S = trapz(x(:),this.M(x, fa, Da, De, ra, re),ndims(x));
            % myfun = @(x) this.M(x, fa, Da, De, ra, re);
            % S = integral(myfun, 0, 1, 'AbsTol', 1e-14, 'ArrayValued', true);

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

            % Initialize the parameters for the first fully connect operation. The first fully connect operation has two input channels.
            parameters = struct;

            parameters.fa   = gpuArray( dlarray(rand([1 img_size],'single') ));
            parameters.Da   = gpuArray( dlarray(rand([1 img_size],'single') )) ;
            parameters.De   = gpuArray( dlarray(rand([1 img_size],'single') )) ;
            parameters.ra   = gpuArray( dlarray(rand([1 img_size],'single') ));
            parameters.fdot = gpuArray( dlarray(rand([1 img_size],'single') ));
            
        end
    
    end

end