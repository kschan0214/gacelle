addpath(genpath('/autofs/space/linen_001/users/kwokshing/tools/askadam/'));
clear;
%% Simulate data

seed        = 8715;
Nsample     = 1e3;
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

rng(seed);

objGPU = gpuAxCaliberSMT(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);

axonDia_range   = [0.1 6];
f_range         = [0, 1];
fscf_range      = [0 0.3];
DeR_range       = [0.5 1.5];
% noise_range     = [0.01 0.05];

axonDia_GT   = single(rand(1,Nsample) * diff(axonDia_range)  + min(axonDia_range) );
fcsf_GT      = single(rand(1,Nsample) * diff(fscf_range)     + min(fscf_range));
f_GT         = single(rand(1,Nsample) * diff(f_range)        + min(f_range));
DeR_GT       = single(rand(1,Nsample) * diff(DeR_range)      + min(DeR_range));

model       = 'VanGelderen';
pars        = [];
pars.a      = single(axonDia_GT);
pars.f      = single(f_GT);
pars.fcsf   = single(fcsf_GT);
pars.DeR    = single(DeR_GT);
s = objGPU.FWD(pars, model);

% Let assume Gaussian noise to simplify everything
noiseLv = 1/SNR;
s       = s + randn(size(s)) .* noiseLv;
s       = permute(s,[2 3 4 1]);
mask    = ones(size(s,1:3));

%% askAdam estimation
fitting                     = [];
fitting.Nepoch              = 4000;
fitting.initialLearnRate    = 0.001;
fitting.decayRate           = 0;
fitting.convergenceValue    = 1e-8;
fitting.tol                 = 1e-3;
fitting.display             = false;
fitting.lambda              = 0;
fitting.isPrior             = 0;

extraData           = [];

out   = objGPU.estimate(s, mask, extraData, fitting);
