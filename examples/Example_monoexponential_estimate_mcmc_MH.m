addpath('/autofs/space/linen_001/users/kwokshing/tools/askadam/')
clear

%% generate some signal based on monoexponential decay
% reproducibility
seed = 5438973; rng(seed); gpurng(seed);


% set up estimation parameters; must be the same as in FWD function
model_param = {'S0','R2star','noise'};

% define number of voxels and SNR
Nsample = 50;
SNR     = 100;

mask        = ones(1,Nsample)>0;
t           = linspace(0,40e-3,15); 
% GT
S0          = 1 + randn(1,Nsample)*0.3;
R2star      = 30 + 5*randn(1,Nsample);
% forward signal generation
pars.(model_param{1}) = S0; 
pars.(model_param{2}) = R2star;
S                     = Example_monoexponential_FWD_mcmc(pars,t);

% realistic signal with certain SNR
noise   = mean(S0) / SNR;           % estimate noise level
y       = S + noise*randn(size(S)); % add Gaussian noise

%% set up fitting algorithm
% set up starting point
pars0.(model_param{1}) = 1 + randn(1,Nsample)*0.3;  % S0
pars0.(model_param{2}) = 30 + 5*randn(1,Nsample);   % R2*
pars0.(model_param{3}) = ones(1,Nsample)*0.001;     % noise

% set up fitting algorithm
fitting                     = [];
% define model parameter name and fitting boundary
fitting.model_params        = model_param;
fitting.lb                  = [0, 0, 0.001];    % lower bound 
fitting.ub                  = [2, 50, 0.1];     % upper bound
fitting.xStepSize           = [0.02, 0.1, 0.05];% step size of proposal for all parameters
% Estimation algorithm setting
fitting.iteration    = 1e5;
fitting.burnin       = 0.1;     % 10% iterations
fitting.thinning     = 5;

% define your forward model
modelFWD = @Example_monoexponential_FWD_mcmc;

% equal weights
weights = [];

mcmc_obj    = mcmc;
xPosterior  = mcmc_obj.metropolis_hastings(y,pars0,weights,fitting,modelFWD,t);

%% plot the estimation results

% compute the mean values of the posterior distribution
S0_mean     = mean(reshape(xPosterior.S0    ,[Nsample, prod(size(xPosterior.S0,2:3))]),2);
R2star_mean = mean(reshape(xPosterior.R2star,[Nsample, prod(size(xPosterior.R2star,2:3))]),2);

figure;
nexttile;scatter(S0,pars0.(model_param{1}));hold on; scatter(S0,S0_mean);refline(1);
xlabel('GT'); ylabel('S0')
nexttile;scatter(R2star,pars0.(model_param{2}));hold on; scatter(R2star,R2star_mean);refline(1)
xlabel('GT'); ylabel('R2*')
legend('Start','fitted')
