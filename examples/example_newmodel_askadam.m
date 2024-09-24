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
mask = ones(1,Nsample);

%% set up fitting algorithm
% set up estimation parameters
model_param = {'S0','R2','noise'};
% set up starting point
x0.(model_param{1}) = 1 + randn(1,Nsample)*0.1;
x0.(model_param{2}) = 20 + 5*randn(1,Nsample);
x0.(model_param{3}) = ones(1,Nsample)*0.001;

weights = ones(size(y));

fitting                     = [];
fitting.Nepoch              = 4000;
fitting.initialLearnRate    = 0.001;
fitting.convergenceValue    = 1e-8;
fitting.tol                 = 1e-3;
fitting.display             = false;
fitting.lambda              = 0;
fitting.lossFunction        = 'l1';
fitting.lambda              = {0};
fitting.randomness = 0;
fitting.model_params = model_param;
fitting.isdisplay = 0;
fitting.convergenceWindow = 20;
fitting.lb           = [0, 0, 0.001];
fitting.ub           = [2, 50, 0.1];

modelFWD = @Example_newmodel_FWD_askadam;

askadam_obj    = askadam;
out  = askadam_obj.optimisation(y,mask,weights,x0,fitting,modelFWD,t);

%% plot the estimation results

figure;
nexttile;scatter(s0,x0.(model_param{1}));hold on; scatter(s0,out.final.S0);refline(1)
nexttile;scatter(R2,x0.(model_param{2}));hold on; scatter(R2,out.final.R2);refline(1)
legend('Start','fitted')
