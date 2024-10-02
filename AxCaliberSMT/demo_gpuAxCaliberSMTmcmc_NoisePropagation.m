addpath(genpath('../askadam/'));
clear;
%% Simulate data

% for reproducibility
seed        = 8715; rng(seed); gpurng(seed);
Nsample     = 1e3;  % #voxel
SNR         = 50;   

% fixed parameters
D0          = 1.7;
Da_fixed    = 1.7;
DeL_fixed   = 1.7;
Dcsf        = 3;

% get current DWI protocol for simulation
bval_sorted     = [0.05, 0.35, 0.80, 1.5, 2.401, 3.45, 4.75, 6, 0.2, 0.95, 2.3, 4.25, 6.75, 9.85, 13.5, 17.8];
ldelta_sorted   = ones(size(bval_sorted))* 6; % ms
BDELTA_sorted   = [13,13,13,13,13,13,13,13,30,30,30,30,30,30,30,30]; %ms

% Parameter raneg for forward simulation
axonDia_range   = [0.1 6];
f_range         = [0.3, 1];
fscf_range      = [0 0.3];
DeR_range       = [0.5 1.5];

% generate ground truth
axonDia_GT   = single(rand(1,Nsample) * diff(axonDia_range)  + min(axonDia_range) );
fcsf_GT      = single(rand(1,Nsample) * diff(fscf_range)     + min(fscf_range));
f_GT         = single(rand(1,Nsample) * diff(f_range)        + min(f_range));
DeR_GT       = single(rand(1,Nsample) * diff(DeR_range)      + min(DeR_range));

% Forward signal simulation
model       = 'VanGelderen';
pars        = [];
pars.a      = axonDia_GT;
pars.f      = f_GT;
pars.fcsf   = fcsf_GT;
pars.DeR    = DeR_GT;
objGPU      = gpuAxCaliberSMTmcmc(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);
s           = objGPU.FWD(pars, model);

% Let assume Gaussian noise for simplicity
noiseLv = 1/SNR;
s       = s + randn(size(s)) .* noiseLv;
s       = gather(permute(s,[2 3 4 1]));
mask    = ones(size(s,1:3))>0;    % create mask

%% MCMC estimation
fitting             = [];
fitting.algorithm   = 'GW';
fitting.Nwalker     = 50;
fitting.StepSize    = 2;
fitting.iteration   = 1e4;
fitting.thinning    = 10;        % Sample every 20 iteration
fitting.metric      = {'median','iqr'};
fitting.burnin      = 0.1;       % 10% burn-in
extraData           = [];

out   = objGPU.estimate(s, mask, extraData, fitting);

%% plot result
field = fieldnames(pars);
tiledlayout(1,numel(field));
for k = 1:numel(field)
    nexttile;
    scatter(pars.(field{k}),out.median.(field{k}),5,'filled','MarkerFaceAlpha',.4);
    h = refline(1);
    h.Color = 'k';
    title(field{k});
    xlabel('GT');ylabel('Estimate');
end