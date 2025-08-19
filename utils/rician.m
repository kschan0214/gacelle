classdef rician < handle
% functions related to rician noise
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 23 June 2025
% Date modified: 

    properties
    end

    properties (GetAccess = public, SetAccess = protected)

    end
    
    methods

        % % constructuor
        % function this = rician()
        % 
        % 
        % end


    end

    methods(Static)

        % for normal usage
        % first moment of Rician distribution
        function S_RM = rician_mean(S,sigma)
            S_RM = max(sigma .* sqrt(pi/2) .* this.L12(-S.^2/2./sigma.^2),S);
        end

        function y = L12(x)
            y = exp(x/2) .* ( (1-x).*besseli(0,-x/2) - x.*besseli(1,-x/2) );
        end

        % for gacelle
        % first moment of Rician distribution
        function S_RM = rician_mean_gacelle(S,sigma)
            S_RM = max(sigma .* sqrt(pi/2) .* rician.L12_gacelle(-S.^2/2./sigma.^2),S);
        end

        function y = L12_gacelle(x)
            x = max(x,-170);
            % TODO: besseli does not support dlarray
            y = exp(x/2) .* ( (1-x).*rician.besseli_gacelle(0,-x/2) - x.*rician.besseli_gacelle(1,-x/2) );
            % y = exp(x/2) .* ( (1-x).*besseli(0,-x/2) - x.*besseli(1,-x/2) );
            % y = hypergeom(-1/2,1,x);
        end

        function I = besseli_gacelle(nu,z)
            Nx  = 34;    % NRMSE<0.05% for Nx=34
            x   = zeros([ones(1,ndims(z)), Nx]); x(:) = linspace(0,pi,Nx);
            I   = 1/pi * trapz(x(:),exp(z.*cos(x)).*cos(nu*x),ndims(x));
        end

        
    end

end