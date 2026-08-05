% Example_monoexponential_FWD_askadam_3D_Strategy1.m
%
% Example: monoexponential decay fitting on a 4D data
% using askadam. Generates GT S0/R2* maps, simulates noisy signal at SNR=100,
% fits with askadam, and plots GT vs. start vs. fitted results.
%
% Kwok-Shing Chan
% Date create: 5 August 2026
%
addpath('../../gacelle/');addpath_gacelle;
clear

%% generate some signal based on monoexponential decay
% reproducibility
seed = 5438973; rng(seed); gpurng(seed);

% set up estimation parameters; must be the same as in FWD function
modelParams = {'S0','R2star'};

% define number of voxels and SNR
Nx      = 21;
Ny      = 21;
Nz      = 21;
SNR     = 100;
% let's create a spherical mask
mask        = strel('sphere',10);mask = mask.Neighborhood;
t           = linspace(0,40e-3,15); 
% GT
S0          = 1 + randn(Nx,Ny,Nz)*0.3;
R2star      = 30 + 5*randn(Nx,Ny,Nz);
% forward signal generation
pars.(modelParams{1}) = S0; 
pars.(modelParams{2}) = R2star;
% S now is a 4D matrix
S                     = Example_monoexponential_FWD_askadam_3D_Strategy1(pars,t);

% realistic signal with certain SNR
noise   = mean(S0(:)) / SNR;        % estimate noise level
y       = S + noise*randn(size(S)); % add Gaussian noise

%% set up fitting algorithm
% set up starting point
pars0.(modelParams{1}) = 1 + randn(Nx,Ny,Nz)*0.5;  % S0
pars0.(modelParams{2}) = 20 + 10*randn(Nx,Ny,Nz);   % R2*

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
fitting.isOptimiseMemory    = false; 

% define your forward model
modelFWD = @Example_monoexponential_FWD_askadam_3D_Strategy1;

% equal weights
weights = [];

out         = askadam().optimisation(y,mask,weights,pars0,fitting,modelFWD,t);

%% plot the estimation results
figure;
nexttile;scatter(S0(mask>0),pars0.(modelParams{1})(mask>0));hold on; scatter(S0(mask>0),out.final.S0(mask>0));refline(1);
xlabel('GT'); ylabel('S0')
nexttile;scatter(R2star(mask>0),pars0.(modelParams{2})(mask>0));hold on; scatter(R2star(mask>0),out.final.R2star(mask>0));refline(1)
xlabel('GT'); ylabel('R2*')
legend('Start','fitted')
figure; tiledlayout(2,3)
nexttile;imshow(S0(:,:,11).*mask(:,:,11),[0 2]);title('S0 GT')
nexttile;imshow(pars0.(modelParams{1})(:,:,11).*mask(:,:,11),[0 2]);title('S0 Start')
nexttile;imshow(out.final.S0(:,:,11),[0 2]);title('S0 Fitted')
nexttile;imshow(R2star(:,:,11).*mask(:,:,11),[10 60]);title('R2* GT')
nexttile;imshow(pars0.(modelParams{2})(:,:,11).*mask(:,:,11),[10 60]);title('R2* Start')
nexttile;imshow(out.final.R2star(:,:,11),[10 60]);title('R2* Fitted')
