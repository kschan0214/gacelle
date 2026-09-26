function [post, out, tRun] = run_chains(y, fitting, FWDfunc, x0, Nchain, seed)
% RUN_CHAINS Run mcmc_bayes with Nchain independent chains per voxel, as
% Nchain copies of every voxel in one GPU call (repetition = 1). Copy k of
% voxel v is column v + (k-1)*Nv. This is equivalent to repetition = Nchain
% with independent starts, but runs the chains in parallel.
%
%   [post, out, tRun] = run_chains(y, fitting, FWDfunc, x0, Nchain, seed)
%
% y         : [Nv, Nm] data
% fitting   : mcmc_bayes fitting structure (full modelParams)
% FWDfunc   : forward model handle, FWDfunc(pars) -> [Nm, Nv]
% x0        : structure of start points, each field [Nv*Nchain, 1] (over-dispersed by the caller)
% post      : structure, each field [Nv, Ns, Nchain] (native space, incl. recovered nuisance fields)
% out       : mcmc_bayes output
% tRun      : wall time of the sampler call (s)
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%
[Nv, Nm] = size(y);
yy   = repmat(y, Nchain, 1);
yy   = reshape(yy, [Nv*Nchain, 1, 1, Nm]);
mask = true(Nv*Nchain, 1, 1);
w    = ones(size(yy), 'single');
fitting.repetition = 1;
rng(seed); parallel.gpu.rng(seed);
t0 = tic;
evalc('out = mcmc_bayes().optimisation(yy, mask, w, x0, fitting, FWDfunc);');
tRun = toc(t0);
fn = fieldnames(out.posterior);
for k = 1:numel(fn)
    x  = out.posterior.(fn{k});                 % [Nv*Nchain, Ns]
    Ns = size(x, 2);
    post.(fn{k}) = permute(reshape(x, Nv, Nchain, Ns), [1 3 2]);
end
end
