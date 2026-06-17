addpath(genpath('../../gacelle/'));
clear;
%% Simulate data

% for reproducibility
seed        = 8715; rng(seed); gpurng(seed);
Nsample     = 1e3;  % #voxel
SNR         = 100;   

% fixed parameters
D0          = 1.7;
Da_fixed    = 1.7;
DeL_fixed   = 1.7;
Dcsf        = 3;

% get current DWI protocol for simulation
bval_sorted     = [0, 0.05, 0.35, 0.80, 1.5, 2.401, 3.45, 4.75, 6, 0.2, 0.95, 2.3, 4.25, 6.75, 9.85, 13.5, 17.8,...
                   0, 0.05, 0.35, 0.80, 1.5, 2.401, 3.45, 4.75, 6, 0.2, 0.95, 2.3, 4.25, 6.75, 9.85, 13.5, 17.8];
ldelta_sorted   = ones(size(bval_sorted))* 6; % ms
BDELTA_sorted   = [13, 13,13,13,13,13,13,13,13,30,30,30,30,30,30,30,30,...
                   13, 13,13,13,13,13,13,13,13,30,30,30,30,30,30,30,30]; %ms

te_sorted       = [51*ones(1,numel(bval_sorted)/2) 92*ones(1,numel(bval_sorted)/2)] * 1e-3;

% Parameter raneg for forward simulation
r_range         = [0.1 5];
f_range         = [0.3, 1];
fscf_range      = [0 0.3];
DeR_range       = [0.5 1.5];
R2a_range       = [6, 12];
k2a_range       = [2,2.5];
R2e_range       = [17, 50];

% generate ground truth
r_GT         = single(rand(1,Nsample) * diff(r_range)        + min(r_range) );
fcsf_GT      = single(rand(1,Nsample) * diff(fscf_range)     + min(fscf_range));
f_GT         = single(rand(1,Nsample) * diff(f_range)        + min(f_range));
DeR_GT       = single(rand(1,Nsample) * diff(DeR_range)      + min(DeR_range));
R2a_GT       = single(rand(1,1)       * diff(R2a_range)      + min(R2a_range));
k2a_GT       = single(rand(1,1)       * diff(k2a_range)      + min(k2a_range));
R2e_GT       = single(rand(1,Nsample) * diff(R2e_range)      + min(R2e_range));

% Forward signal simulation
model       = 'VanGelderen';
pars        = [];
pars.r      = single(r_GT);
pars.f      = single(f_GT);
pars.fcsf   = single(fcsf_GT);
pars.DeR    = single(DeR_GT);
pars.R2e    = single(R2e_GT);
pars.k2a    = single(k2a_GT);
pars.R2a    = single(R2a_GT);
objGPU      = gpuMEAxCaliberSMT(bval_sorted, ldelta_sorted, BDELTA_sorted, te_sorted, [],[]);
s           = objGPU.FWD(pars, model);

% Let assume Gaussian noise for simplicity
noiseLv = 1/SNR;
s       = s + randn(size(s)) .* noiseLv;
s       = permute(s,[2 3 4 1]);
mask    = ones(size(s,1:3))>0;  % create mask

%% askAdam estimation
fitting                     = [];
fitting.solver              = 'mcmc';
fitting                     = objGPU.check_set_default(fitting);
fitting.start               = 'likelihood';
fitting.algorithm           = 'ensemble';
fitting.Nwalker             = 30;
fitting.StepSize            = 2;
fitting.iteration           = 1e4;
fitting.thinning            = 10;        % Sample every 10 iteration
fitting.metric              = {'median','iqr'};
fitting.isFitR2a            = true;
fitting.isFitk2a            = true;
fitting.isFitCSF            = true;
extraData                   = [];

out   = objGPU.estimate(s, mask, fitting);

%% plot result
figure;
field = fieldnames(pars);
tiledlayout(1,numel(field));
for k = 1:numel(field)
    nexttile;
    scatter(pars.(field{k}),out.median.(field{k}),5,'filled','MarkerFaceAlpha',.4);
    h = refline(1);
    h.Color = 'k';
    title(field{k});
    xlabel('GT');ylabel('Fitted');
end