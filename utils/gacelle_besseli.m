% function I = mybesseli(nu,z)
% substitution function of matlab besseli that is compatible with askAdam and mcmc
function I = gacelle_besseli(nu,z,Nx)
    if nargin < 3
        Nx  = 34;    % NRMSE<0.05% for Nx=34
    end
    x   = zeros([ones(1,ndims(z)), Nx]); x(:) = linspace(0,pi,Nx);
    I   = 1/pi * trapz(x(:),exp(z.*cos(x)).*cos(nu*x),ndims(x));
end