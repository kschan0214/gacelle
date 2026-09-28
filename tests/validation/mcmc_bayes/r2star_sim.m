function [y, mask, pars0, fitting, obj, truth] = r2star_sim(Nv, SNR, seed, te)
% R2STAR_SIM Monoexponential (gpuR2starMapping) simulation for mcmc_bayes validation.
%
%   [y, mask, pars0, fitting, obj, truth] = r2star_sim(Nv, SNR, seed)
%   [...] = r2star_sim(Nv, SNR, seed, te)
%
% Ground truth: M0 ~ U(0.9,1.1), R2star ~ U(20,50) 1/s, noise = 1/SNR
% (Gaussian, S0 = 1 reference), te = linspace(0,40e-3,12) s by default.
% Starting points are the gpuR2starMapping defaults [1, 30, 0.05] for all
% voxels, bounds and xStepSize are the class defaults (mcmc solver, so
% 'noise' is sampled).
%
% Output
% ------
% y         : [Nv,1,1,Nte] data
% mask      : [Nv,1,1] true
% pars0     : starting points, fields [Nv,1,1]
% fitting   : mcmc fitting structure (MH, metric mean/std/median); the
%             caller sets iteration/burnin/thinning and the new options
% obj       : gpuR2starMapping object (use @obj.FWD)
% truth     : ground truth, fields 1xNv
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%
if nargin < 4 || isempty(te); te = linspace(0, 40e-3, 12); end

rng(seed);
truth.M0     = 0.9 + 0.2  * rand(1, Nv);
truth.R2star = 20  + 30   * rand(1, Nv);
truth.noise  = ones(1, Nv) / SNR;

obj     = gpuR2starMapping(te);
s       = obj.FWD(truth);                       % [Nte, Nv]
y       = s + truth.noise .* randn(size(s));
y       = permute(y, [2 3 4 1]);                % [Nv,1,1,Nte]
mask    = true(size(y, 1:3));

fitting             = [];
fitting.solver      = 'mcmc';
fitting.algorithm   = 'MH';
fitting.metric      = {'mean','std','median'};
fitting.start       = 'default';
fitting             = obj.check_set_default(fitting);
obj                 = obj.updateProperty(fitting);
fitting.modelParams = obj.modelParams;
fitting.ub          = obj.ub(1:numel(obj.modelParams));
fitting.lb          = obj.lb(1:numel(obj.modelParams));
evalc('pars0 = obj.determine_x0(y, mask, fitting);');
fitting.xStepSize   = obj.step;

end
