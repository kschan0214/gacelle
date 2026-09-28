function s = lingauss_fwd(pars, A)
% LINGAUSS_FWD Linear Gaussian toy forward model for the Phase 3 tests:
%   s = A * [u1; u2; ...; ud],  one column per voxel.
%
%   s = lingauss_fwd(pars, A)
%
% pars  : structure with fields u1..ud, each 1xNv (gpuArray); other fields
%         (e.g. 'noise') are ignored
% A     : [Nm, d] design matrix
% s     : [Nm, Nv], mcmc FWD convention
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%
d = size(A, 2);
u = zeros(d, numel(pars.u1), 'like', pars.u1);
for k = 1:d
    u(k,:) = pars.(sprintf('u%d', k))(:).';
end
s = cast(A, 'like', u) * u;
end
