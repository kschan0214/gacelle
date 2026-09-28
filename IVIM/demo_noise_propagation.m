%% demo_noise_propagation.m
%
% This demo provides examples on the utilisation of gpuIVIM.m for parameter
% estimation with simulated data
%
% Kwok-Shing Chan
% kchan2@mgh.harvard.edu
%
% Date created: 28 September 2026
% Date modified:
%
addpath('../../gacelle'); addpath_gacelle; % this is the path to 'gacelle' package
clear;
%% Simulate data

% for reproducibility
seed        = 23439; rng(seed); gpurng(seed);
Nsample     = 1e3;  % #voxel
SNR         = 50;   % at b0

% IVIM protocol, b-values in ms/um2 (1000 s/mm2 = 1 ms/um2)
b           = [0 10 20 40 80 110 140 170 200 300 400 500 600 700 800 900] / 1000;

% Parameter range for forward simulation
f_range     = [0.05, 0.2];
D_range     = [0.6, 1.2];      % um2/ms
Dstar_range = [10, 40];        % um2/ms

% generate ground truth
pars        = [];
pars.S0     = ones(1,Nsample,'single');
pars.f      = single(rand(1,Nsample) * diff(f_range)     + min(f_range));
pars.D      = single(rand(1,Nsample) * diff(D_range)     + min(D_range));
pars.Dstar  = single(rand(1,Nsample) * diff(Dstar_range) + min(Dstar_range));
objGPU      = gpuIVIM(b);
s           = objGPU.FWD(pars);

% Let's assume Gaussian noise for simplicity
y       = s + randn(size(s)) / SNR;
y       = permute(y,[3 2 4 1]);     % [1, Nsample, 1, Nb]
mask    = ones(size(y,1:3)) > 0;

%% askadam estimation
rng(seed); gpurng(seed);

fitting             = [];
fitting.solver      = 'askadam';
fitting.iteration   = 2000;
out_adam            = gpuIVIM(b).estimate(y, mask, [], fitting);

%% mcmc estimation (a new object per fit: fit() adapts the parameter list to the solver)
rng(seed); gpurng(seed);

fitting             = [];
fitting.solver      = 'mcmc';
fitting.algorithm   = 'MH';
fitting.iteration   = 20000;
fitting.burnin      = 0.5;
fitting.thinning    = 10;
fitting.metric      = {'median','std'};
out_mcmc            = gpuIVIM(b).estimate(y, mask, [], fitting);

%% plot
pn = {'f','D','Dstar'};
figure;
for k = 1:numel(pn)
    subplot(2,3,k);   scatter(pars.(pn{k}), out_adam.final.(pn{k})(:), 5, 'filled'); refline(1,0); title(['askadam: ' pn{k}]); xlabel('truth'); ylabel('estimate');
    subplot(2,3,k+3); scatter(pars.(pn{k}), out_mcmc.median.(pn{k})(:), 5, 'filled'); refline(1,0); title(['mcmc: ' pn{k}]); xlabel('truth'); ylabel('estimate');
end
