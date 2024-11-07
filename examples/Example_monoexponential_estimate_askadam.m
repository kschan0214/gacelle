addpath(genpath('../../gacelle/'))
clear

%% generate some signal based on monoexponential decay
% reproducibility
seed = 5438973; rng(seed); gpurng(seed);

% set up estimation parameters; must be the same as in FWD function
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
S           = Example_monoexponential_FWD_askadam(pars,t);

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
fitting.modelParams         = modelParams;
fitting.lb                  = [0, 0];   % lower bound 
fitting.ub                  = [2, 50];  % upper bound
% Estimation algorithm setting
fitting.iteration           = 4000;
fitting.initialLearnRate    = 0.001;
fitting.lossFunction        = 'l1';
fitting.tol                 = 1e-4;
fitting.convergenceValue    = 1e-8;
fitting.convergenceWindow   = 20;
fitting.isDisplay          = false;

% define your forward model
modelFWD = @Example_monoexponential_FWD_askadam;

% equal weights
weights = [];

askadam_obj = askadam;
out         = askadam_obj.optimisation(y,mask,weights,pars0,fitting,modelFWD,t);

%% plot the estimation results
figure;
nexttile;scatter(S0,pars0.(modelParams{1}));hold on; scatter(S0,out.final.S0);refline(1);
xlabel('GT'); ylabel('S0')
nexttile;scatter(R2star,pars0.(modelParams{2}));hold on; scatter(R2star,out.final.R2star);refline(1)
xlabel('GT'); ylabel('R2*')
legend('Start','fitted')
