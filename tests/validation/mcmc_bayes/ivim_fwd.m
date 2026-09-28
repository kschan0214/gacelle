function s = ivim_fwd(pars, b, solver, fitting)
% IVIM_FWD Minimal bi-exponential IVIM forward model for mcmc_bayes validation.
%
%   S = S0 * [ F*exp(-b*Dstar) + (1-F)*exp(-b*D) ]
%
%   s = ivim_fwd(pars, b)
%   s = ivim_fwd(pars, b, solver, fitting)
%
% Input
% -----
% pars      : structure, each field 1xNv (gpuArray)
%   .S0         : (optional) amplitude, default 1
%   .D          : diffusion coefficient, same unit as 1/b
%   .F          : perfusion fraction [0,1]
%   .Dstar      : pseudo-diffusion coefficient, same unit as 1/b
% b         : b-values, Nb elements
% solver    : (optional) unused, kept for the mcmc FWD signature
% fitting   : (optional) fitting structure; with fitting.algorithm =
%             'ensemble' the output is reshaped to [Nb, Nv/Nwalker, Nwalker]
%
% Output
% ------
% s         : [Nb, Nv] signal, following the mcmc FWD convention
%             (measurements in rows, voxels in columns)
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 26 September 2026
%

if nargin < 4; fitting = []; end

b = cast(b(:), 'like', pars.D);    % [Nb,1]

D       = pars.D(:).';              % [1,Nv]
F       = pars.F(:).';
Dstar   = pars.Dstar(:).';

s = F .* exp(-b .* Dstar) + (1 - F) .* exp(-b .* D);

if isfield(pars, 'S0') && ~isempty(pars.S0)
    s = pars.S0(:).' .* s;
end

% reshape s for ensemble solver
if ~isempty(fitting) && isfield(fitting, 'algorithm') && any(strcmpi(fitting.algorithm, {'ensemble','gw'}))
    s = reshape(s, [size(s,1) size(s,2)/fitting.Nwalker fitting.Nwalker]);
end

end
