function y = ivim_sim(cfg, SNR, Nrep, seed, noiseType)
% IVIM_SIM Simulate IVIM signals for the ground-truth voxels in cfg (ivim_t2_config).
%
%   y = ivim_sim(cfg, SNR, Nrep, seed, noiseType)
%
% SNR       : S0/sigma, sigma = S0_truth/SNR per voxel
% Nrep      : # noise realisations per truth voxel (voxel order: truth index fastest)
% noiseType : 'gaussian' | 'rician' (magnitude of complex Gaussian noise)
% y         : [Nv*Nrep, Nb] double
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%
if nargin < 5; noiseType = 'gaussian'; end
rng(seed);
Nv  = numel(cfg.truth.D);
p.S0 = cfg.truth.S0; p.D = cfg.truth.D; p.F = cfg.truth.F; p.Dstar = cfg.truth.Dstar;
s   = ivim_fwd(p, cfg.b).';                 % [Nv, Nb]
s   = repmat(s, Nrep, 1);                   % truth index fastest
sig = repmat(cfg.truth.S0(:)/SNR, Nrep, 1);
switch lower(noiseType)
    case 'gaussian'
        y = s + sig.*randn(size(s));
    case 'rician'
        y = sqrt((s + sig.*randn(size(s))).^2 + (sig.*randn(size(s))).^2);
    otherwise
        error('unknown noiseType %s', noiseType);
end
end
