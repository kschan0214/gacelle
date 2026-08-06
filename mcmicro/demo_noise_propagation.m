%% demo_noise_propagation.m
%
% This demo provides several examples on the ulitisation of gpumcmicro.m 
% for parameter estimation with simulated data
% 
% Kwok-Shing Chan 
% kchan2@mgh.harvard.edu
%
% Date created: 5 August 2026 
% Date modified: 
%
addpath('../../gacelle'); addpath_gacelle; % this is the path to 'gacelle' package
clear;
%% Simulate data

% for reproducibility
seed        = 23439; rng(seed); gpurng(seed);
Nsample     = 1e3;  % #voxel
SNR         = 40;  % at b0

% get current DWI protocol for simulation
bval_sorted     = [0, 0.5, 1, 2];

% Parameter raneg for forward simulation
f_range    = [0.1, 0.8];
D_range    = [1.5, 3];

% generate ground truth
D_GT        = single(rand(1,Nsample) * diff(D_range)  + min(D_range));
f_GT        = single(rand(1,Nsample) * diff(f_range)  + min(f_range));

pars        = [];
pars.f      = f_GT;
pars.D      = D_GT;
objGPU      = gpumcmicro(bval_sorted);
s           = objGPU.FWD(pars);

% Let's assume Gaussian noise for simplicity
noiseLv = 1/SNR;
y       = s + randn(size(s)) .* noiseLv;
y       = permute(y,[3 2 4 1]);
mask    = ones(size(y,1:3)) > 0;

%% askadam estimation
rng(seed); gpurng(seed);

fitting             = [];
fitting.solver      = 'askadam';
fitting             = objGPU.check_set_default(fitting);
out_adam            = objGPU.estimate(y, mask, fitting);

% get initial starting point based on likelihood method for scatter plots
rng(seed); gpurng(seed);
fitting.iteration   = 0;
[~,pars0] = evalc('objGPU.estimate( y, mask, fitting)');

%% mcmc estimation
% reset class object
objGPU      = gpumcmicro(bval_sorted);

rng(seed); gpurng(seed);

fitting         = [];
fitting.solver  = 'mcmc';
fitting         = objGPU.check_set_default(fitting);
out_mcmc        = objGPU.estimate(y, mask, fitting);

%% make some plots

% get comparable residuals on mcmc's mean posterior
fitting             = [];
fitting.solver      = 'askadam';
fitting.iteration   = 0;
fitting             = objGPU.check_set_default(fitting);
[~,loss_mcmc] = evalc('objGPU.estimate( y, mask, fitting, [], out_mcmc.mean)');

% plot result
field = fieldnames(pars);
figure;tiledlayout(2,numel(field),"TileSpacing","compact");
for k = 1:numel(field)
    nexttile;
    scatter(pars.(field{k}),pars0.final.(field{k}),5,'filled','MarkerFaceAlpha',.4);hold on
    scatter(pars.(field{k}),out_adam.final.(field{k}),5,'filled','MarkerFaceAlpha',.4);
    scatter(pars.(field{k}),out_mcmc.mean.(field{k}),5,'filled','MarkerFaceAlpha',.4);
    h = refline(1);
    h.Color = 'k';
    title(field{k});
    xlabel('GT');ylabel('Fitted');
end
xlabel('GT');ylabel('Fitted');
legend('Start','askadam.m estimation','mcmc estimation')

nexttile([1 numel(field)]);plot(pars0.final.resloss,'x');hold on;plot(out_adam.final.resloss,'o');plot(loss_mcmc.final.resloss,'o');
title('Residuals on each sample'); 
legend('Start','askadam.m estimation','mcmc.m estimation');xlabel('Sample');ylabel('residual');

