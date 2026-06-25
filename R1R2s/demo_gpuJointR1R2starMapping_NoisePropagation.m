
addpath(genpath('../../gacelle/'));
clear;

%% Simulate data

% for reproducibility
seed        = 23439; rng(seed);
Nsample     = 1e3;  % #voxel
SNR         = 100;

Nt  = 15; 
TE1 = 1.5e-3;
tr  = 45e-3;
t   = linspace(TE1, tr-3e-3, Nt);
fa  = [5, 10, 20, 30, 40, 50, 70];
Nfa = numel(fa);

% Parameter raneg for forward simulation
M0_range        = [0.5, 2];
R1_range        = [0.2, 2.5];
R2star_range    = [5,   60];

% generate ground truth
M0_GT       = single(rand(1,Nsample) * diff(M0_range) + min(M0_range) );
R1_GT       = single(rand(1,Nsample) * diff(R1_range)  + min(R1_range));
R2star_GT   = single(rand(1,Nsample) * diff(R2star_range)  + min(R2star_range));

% forward signal simulation
pars.M0     = M0_GT;
pars.R1     = R1_GT;
pars.R2star = R2star_GT;

b1      = ones(size(M0_GT)); extraData.b1 = b1;
objGPU  = gpuJointR1R2starMapping(t,tr,fa);
s       = gather((objGPU.FWD(pars, extraData)));
s       = permute(reshape(s,[Nt, Nfa, Nsample]),[3 4 5 1 2]);
mask    = ones(size(s,1:3),'logical');

% Let's assume Gaussian noise for simplicity
noiseLv = (M0_GT.')./SNR;
s       = s + randn(size(s)) .* noiseLv;

extraData                   = [];
extraData.b1                = b1.';

%% DEMO#1: askadam estimation default
fitting             = [];
fitting.solver      = 'askadam';
fitting             = objGPU.check_set_default(fitting);
fitting.start       = 'default';

objGPU              = gpuJointR1R2starMapping(t,tr,fa);
out_adam            = objGPU.estimate(s, mask, extraData, fitting);

% get starting position
fitting.iteration   = 0;
pars0               = objGPU.estimate(s, mask, extraData,fitting);

%% DEMO#2: mcmc estimation, uniform weights
% reset class object
objGPU              = gpuJointR1R2starMapping(t,tr,fa);

fitting             = [];
fitting.solver      = 'mcmc';
fitting             = objGPU.check_set_default(fitting);

out_mh              = objGPU.estimate(s, mask, extraData, fitting);

%% plot result
figure(99);
field = fieldnames(pars);
tiledlayout(2,numel(field));
for k = 1:numel(field)
    nexttile;
    scatter(pars.(field{k}),pars0.final.(field{k}),5,'filled','MarkerFaceAlpha',.4);hold on
    scatter(pars.(field{k}),out_adam.final.(field{k}),5,'filled','MarkerFaceAlpha',.4);
    h = refline(1);
    h.Color = 'k';
    title(field{k});
    xlabel('GT');ylabel('Fitted');
end
legend('Start','askadam.m estimation','Location','northwest');

for k = 1:numel(field)
    nexttile;
    scatter(pars.(field{k}),out_mh.mean.(field{k}),5,'filled','MarkerFaceAlpha',.4,'MarkerFaceColor','g');
    h = refline(1);
    h.Color = 'k';
    title(field{k});
    xlabel('GT');ylabel('Fitted');
end
legend('mcmc.m estimation','Location','northwest');