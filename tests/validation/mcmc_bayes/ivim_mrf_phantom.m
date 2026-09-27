function [y, mask, truth, f] = ivim_mrf_phantom(SNR, seed)
% IVIM_MRF_PHANTOM Small piecewise-smooth IVIM phantom for the Phase 4 MRF scripts (T4.5, T4.6).
%
%   [y, mask, truth, f] = ivim_mrf_phantom(SNR, seed)
%
% Geometry: 16 x 16 x 4, mask = disc of radius 7.5 voxels in every slice (~700 voxels).
% Classes [D um^2/ms, F, Dstar um^2/ms]: outer ring [0.8 0.05 15], inner disc (radius < 4)
%   [1.0 0.10 25], lesion (3 x 3 x 2 block in the ring) [1.4 0.02 10]. Within-class variation:
%   D and Dstar x exp(0.05 randn), F + 0.01 randn (clamped to [0.005, 0.5]); S0 = 1 + 0.05 randn.
% Noise: Gaussian, sigma = 1/SNR. b-values of ivim_t2_config.m (0..900 s/mm^2, ms/um^2 units).
%
% y     : [16 16 4 Nb] data
% mask  : [16 16 4] logical
% truth : structure of images S0, D, F, Dstar, and the class label image
% f     : mcmc_bayes fitting structure (full modelParams S0 D F Dstar noise; 'marginal_S0noise',
%         S0Param 'S0'; transforms linear/log/sigmoid/log/linear with lb = 0, ub = Inf for D and
%         Dstar, F in [0, 1], so that all three are allowed under the hierarchical prior)
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%
rng(seed);
dims = [16 16 4];
cfg  = ivim_t2_config();
[x1, x2] = ndgrid((1:dims(1)) - 8.5, (1:dims(2)) - 8.5);
rr   = sqrt(x1.^2 + x2.^2);
mask = repmat(rr <= 7.5, 1, 1, dims(3));
cls  = repmat(1 + (rr < 4), 1, 1, dims(3));             % 1 ring, 2 inner disc
cls(2:4, 7:9, 2:3) = 3;                                 % lesion in the ring
cls(~mask) = 0;
val  = [0.8 0.05 15; 1.0 0.10 25; 1.4 0.02 10];
truth.class = cls;
truth.D = zeros(dims); truth.F = zeros(dims); truth.Dstar = zeros(dims);
for c = 1:3
    ic = cls == c; n = nnz(ic);
    truth.D(ic)     = val(c,1) .* exp(0.05*randn(n,1));
    truth.F(ic)     = min(max(val(c,2) + 0.01*randn(n,1), 0.005), 0.5);
    truth.Dstar(ic) = val(c,3) .* exp(0.05*randn(n,1));
end
truth.S0 = (1 + 0.05*randn(dims)) .* mask;
p.S0 = truth.S0(mask).'; p.D = truth.D(mask).'; p.F = truth.F(mask).'; p.Dstar = truth.Dstar(mask).';
s    = ivim_fwd(p, cfg.b);                              % [Nb, Nv]
yv   = s + randn(size(s))/SNR;
y    = zeros(numel(mask), numel(cfg.b));
y(mask(:), :) = yv.';
y    = reshape(y, [dims numel(cfg.b)]);

f.modelParams = {'S0';'D';'F';'Dstar';'noise'};
f.lb = [0; 0; 0; 0; 0.001]; f.ub = [2; Inf; 1; Inf; 1];
f.xStepSize = [0.01; 0.05; 0.01; 2; 0.001];
f.parameterTransform = {'linear','log','sigmoid','log','linear'};
f.likelihood = 'marginal_S0noise'; f.S0Param = 'S0';
f.algorithm = 'MH'; f.metric = {'mean'}; f.repetition = 1;
f.adaptStepSize = true; f.adaptInterval = 50; f.updateScheme = 'joint';
f.b = cfg.b;
end
