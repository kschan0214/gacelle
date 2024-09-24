addpath('/autofs/space/linen_001/users/kwokshing/tools/askadam/')
clear

% reproducibility
seed = 5438973;
rng(seed); gpurng(seed);

%% generate some signal based on monoexponential decay
Nsample = 50;

t = linspace(0,40e-3,15); t = t(:);
s0 = 1 + randn(1,Nsample)*0.1;
R2 = 20 + 5*randn(1,Nsample);
xGT.S0 = s0;
xGT.R2 = R2;
s = Example_newmodel_FWD(xGT,t);
y = s;

%% set up fitting algorithm
% set up estimation parameters
model_param = {'S0','R2','noise'};
% set up starting point
x0.(model_param{1}) = 1 + randn(1,Nsample)*0.1;
x0.(model_param{2}) = 20 + 5*randn(1,Nsample);
x0.(model_param{3}) = ones(1,Nsample)*0.001;

weights = ones(size(y));

fitting              = [];
fitting.model_params = model_param;
fitting.burnin       = 0.1; 
fitting.repetition   = 1;
fitting.iteration    = 1e4;
fitting.thinning     = 5;
fitting.StepSize     = 2;
fitting.Nwalker      = 50;
fitting.lb           = [0, 0, 0.001];
fitting.ub           = [2, 50, 0.1];

modelFWD = @Example_newmodel_FWD;

mcmc_obj    = mcmc;
xPosterior  = mcmc_obj.goodman_weare(y,x0,weights,fitting,modelFWD,t);

%% plot the estimation results
S0_mean = mean(reshape(xPosterior.S0 ,[Nsample, prod(size(xPosterior.S0,2:3))]),2);
R2_mean = mean(reshape(xPosterior.R2 ,[Nsample, prod(size(xPosterior.R2,2:3))]),2);

nexttile;scatter(s0,x0.(model_param{1}));hold on; scatter(s0,S0_mean);refline(1)
nexttile;scatter(R2,x0.(model_param{2}));hold on; scatter(R2,R2_mean);refline(1)
legend('Start','Mean posterior')
