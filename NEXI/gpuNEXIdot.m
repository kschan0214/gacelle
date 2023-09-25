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
%  Authors: 
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

        function [fa, Da, De, ra, fdot] = fit(this,y)

            % get model parameters
            parameters = this.initialise_model(NVoxel);
    
            % clear cache before running everthing
            accfun = dlaccelerate(@this.modelGradients);
            clearCache(accfun)
            
            numEpochs = 4000;
    
            % optimisation process
            averageGrad     = [];
            averageSqGrad   = [];
            % optimisation
            for epoch = 1:numEpochs
                    
                % Evaluate the model gradients and loss using dlfeval and the modelGradients function.
                [gradients,loss] = dlfeval(accfun,parameters,y);
            
                % Update learning rate.
                learningRate = initialLearnRate / (1+decayRate*epoch);
            
                % Update the network parameters using the adamupdate function.
                [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
                    averageSqGrad,epoch,learningRate);
            
                    
                % Plot training progress.
                loss = double(gather(extractdata(loss)));
                if isdisplay
                    addpoints(lineLoss,epoch, loss);
                
                    D = duration(0,0,toc(start),'Format','hh:mm:ss');
                    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
                    drawnow
                end
                
                if mod(epoch,100) == 0 && isdisplay
                    % print intermediate
                    rse     = sqrt(sum(gather(extractdata( ((abs(img - model_singlecompartment_r1r2s(parameters,te,fa_map,tr))).^2))),'all','omitnan'));
                    NRSE    =  rse / gather(extractdata(sqrt(sum(abs(img).^2,'all'))));
                    fprintf('#Epoch: %i, NRSE: %f \n',epoch,NRSE);
                end
            
                % save intermediate results
                if mod(epoch,250) == 0 && isSaveIntermediate
                    dloptimiser.parameters    = parameters;
                    dloptimiser.epoch         = epoch;
                    save(fullfile(output_dir,[output_filename '_epoch_' num2str(epoch)]),'dloptimiser');
                    export_fig(fullfile(output_dir,[output_filename '_loss_' num2str(epoch)]),'-png');
                end
            end

            fa      = parameters.fa;
            Da      = parameters.Da;
            De      = parameters.De;
            ra      = parameters.ra;
            fdot    = parameters.fdot;

        end

        function [gradients,loss] = modelGradients(this, parameters, dlR)

            % Make predictions with the initial conditions.
            % parameters.r2s = parameters.r2s * 10;   % upscale R2* to the usualy range
            R = this.FWD(parameters);
            % parameters.r2s = parameters.r2s / 10;   % downscale R2* to the optimisation range
            
            R   = dlarray(R(:).',     'CB');
            dlR = dlarray(dlR(:).',   'CB');
            
            % compute MSE
            loss = mse(R, dlR);
            
            % Calculate gradients with respect to the learnable parameters.
            gradients = dlgradient(loss,parameters);
        
        end

        function [s] = FWD(this, pars)
                fa   = pars.fa(:,1);
                Da   = pars.Da(:,2);
                De   = pars.De(:,3);
                ra   = pars.ra(:,4);
                fdot = pars.fdot(:,5);
                
                % Forward model
                S = this.S(fa, Da, De, ra);
                
                % Combined signal
                s = (1-fdot).*S + fdot;
                           
        end

        function S = S(this, fa, Da, De, ra)
            Da = this.b*Da;
            De = this.b*De;
            ra = ra*this.Delta;
            re = ra*fa/(1-fa);
            
            % Trapezoidal's rule
            x = linspace(0,1,3);
            S = trapz(x,this.M(x, fa, Da, De, ra, re),2);
            
            % myfun = @(x) this.M(x, fa, Da, De, ra, re);
            % S = integral(myfun, 0, 1, 'AbsTol', 1e-14, 'ArrayValued', true);
        end
    end

    methods(Static)
        function M = M(x, fa, Da, d2, r1, r2)
            % d1 = b*Da*x^2
            % d2 = b*De
            % r1 = ra*t
            % r2 = ra*t*fa/(1-fa)
            d1 = Da.*x.^2;
            l1 = (r1+r2+d1+d2)/2;
            l2 = sqrt( (r1-r2+d1-d2).^2 + 4*r1.*r2 )/2;
            lm = l1-l2;
            Pp = (fa*d1 + (1-fa)*d2 - lm)./(l2*2);
            M  = Pp.*exp(-(l1+l2)) + (1-Pp).*exp(-lm); 
        end

        function parameters = initialise_model(NVoxel)

            % Initialize the parameters for the first fully connect operation. The first fully connect operation has two input channels.
            parameters = struct;
            
            % assume comparment a has long T1 and T2*
            % it is important to have all parameters in the same range to get similar
            % stepp size
            parameters.fa   = gpuArray( dlarray(rand(NVoxel,1,'single') )) ;
            parameters.Da   = gpuArray( dlarray(rand(NVoxel,1,'single') )) ;
            parameters.De   = gpuArray( dlarray(rand(NVoxel,1,'single') )) ;
            parameters.ra   = gpuArray( dlarray(rand(NVoxel,1,'single') )) ;
            parameters.fdot = gpuArray( dlarray(rand(NVoxel,1,'single') )) ;
            
        end
    
        
    
    end

end