% Example_monoexponential_estimate_askadam.m
%
% Minimal, direct use of askadam.optimisation on vectorised (GACELLE
% dimension, 'G-D') input: a two-parameter monoexponential decay (S0, R2*) is
% simulated for Nsample independent voxels, Gaussian noise is added at a
% fixed SNR, and the parameters are recovered by calling askadam directly
% against a user-supplied forward function, without going through any of
% GACELLE's built-in model classes (e.g. gpuNEXI, gpuAxCaliberSMT). Useful as
% a template for fitting a custom forward model outside the model-class
% framework. Recovered values are plotted against ground truth and the
% random starting points.
%
% Kwok-Shing Chan
% Date create: 5 August 2026
%
%% add path
addpath('../../gacelle/');addpath_gacelle;
clear

%% generate some signal based on monoexponential decay
% reproducibility
rng('default'); seed = 5438973; rng(seed); gpurng(seed);

% set up estimation parameter name; must be the same as the fields of 'pars' in the forward function
modelParams = {'S0','R2star'};

% define number of voxels and SNR
Nsample = 100;
SNR     = 100;

mask        = ones(1,Nsample)>0;
t           = linspace(0,40e-3,15); 
% GT
S0          = 1 + randn(1,Nsample)*0.3;
R2star      = 30 + 5*randn(1,Nsample);
% forward signal generation
pars.(modelParams{1}) = S0; 
pars.(modelParams{2}) = R2star;
S = Example_monoexponential_FWD_GD(pars,t);

% realistic signal with certain SNR
noise   = mean(S0) / SNR;           % estimate noise level
y       = S + noise*randn(size(S)); % add Gaussian noise

%% set up fitting algorithm
% set up starting point
pars0.(modelParams{1}) = 1 + randn(1,Nsample)*0.5;  % S0
pars0.(modelParams{2}) = 20 + 10*randn(1,Nsample);   % R2*

% set up fitting algorithm
fitting                     = [];
% define model parameter name and fitting boundary
fitting.modelParams         = {'S0','R2star'}; % modelParams;
fitting.lb                  = [0, 0];   % lower bound 
fitting.ub                  = [2, 50];  % upper bound
% Estimation algorithm setting
fitting.iteration           = 10000;
fitting.initialLearnRate    = 0.01;
fitting.decayRate           = 0.001;
fitting.lossFunction        = 'l1';

% define your forward model
modelFWD = @Example_monoexponential_FWD_GD;

% equal weights
weights = [];

out     = askadam().optimisation(y,mask,weights,pars0,fitting,modelFWD,t);

%% plot the estimation results
figure;
nexttile;scatter(S0,pars0.(modelParams{1}));hold on; scatter(S0,out.final.S0);refline(1);
xlabel('GT'); ylabel('S0')
nexttile;scatter(R2star,pars0.(modelParams{2}));hold on; scatter(R2star,out.final.R2star);refline(1)
xlabel('GT'); ylabel('R2*')
legend('Start','fitted')
