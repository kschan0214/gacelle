addpath('/autofs/space/linen_001/users/kwokshing/tools/askadam/')
clear

%% generate some signal based on monoexponential decay
% reproducibility
seed = 5438973; rng(seed); gpurng(seed);


% set up estimation parameters; must be the same as in FWD function
modelParams = {'S0','R2star','noise'};

% define number of voxels and SNR
Nsample = 50;
SNR     = 100;

mask        = ones(1,Nsample)>0;
t           = linspace(0,40e-3,15); 
% GT
S0          = 1 + randn(1,Nsample)*0.3;
R2star      = 30 + 5*randn(1,Nsample);
% forward signal generation
pars.(modelParams{1}) = S0; 
pars.(modelParams{2}) = R2star;
S                     = Example_monoexponential_FWD_mcmc(pars,t);

% realistic signal with certain SNR
noise   = mean(S0) / SNR;           % estimate noise level
y       = S + noise*randn(size(S)); % add Gaussian noise
y       = permute(y, [2 3 4 1]);    % make sure spatial info in the first 3 dimensions while measurements in 4th dimension

%% set up fitting algorithm
% set up starting point
pars0.(modelParams{1}) = 1 + randn(1,Nsample)*0.3;  % S0
pars0.(modelParams{2}) = 30 + 5*randn(1,Nsample);   % R2*
pars0.(modelParams{3}) = ones(1,Nsample)*0.001;     % noise

% set up fitting algorithm
fitting                     = [];
% define model parameter name and fitting boundary
fitting.modelParams        = modelParams;
fitting.lb                  = [0, 0, 0.001];   % lower bound 
fitting.ub                  = [2, 50, 0.1];  % upper bound
% Estimation algorithm setting
fitting.iteration    = 1e4;
fitting.algorithm    = 'GW';
fitting.burnin       = 0.1;     % 10% iterations
fitting.thinning     = 5;
fitting.StepSize     = 2;
fitting.Nwalker      = 50;

% define your forward model
modelFWD = @Example_monoexponential_FWD_mcmc;

% equal weights
weights = [];

mcmc_obj    = mcmc;
out         = mcmc_obj.optimisation(y,mask,weights,pars0,fitting,modelFWD,t);

%% plot the estimation results

figure;
nexttile;scatter(S0,pars0.(modelParams{1}));hold on; scatter(S0,out.mean.S0);refline(1);
xlabel('GT'); ylabel('S0')
nexttile;scatter(R2star,pars0.(modelParams{2}));hold on; scatter(R2star,out.mean.R2star);refline(1)
xlabel('GT'); ylabel('R2*')
legend('Start','fitted')
