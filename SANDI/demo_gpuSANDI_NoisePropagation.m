addpath(genpath('../../gacelle/'));
clear;
%% Simulate data

% for reproducibility
seed        = 23439; rng(seed); gpurng(seed);
Nsample     = 1e3;  % #voxel
SNR         = 50;  % at b0
Ds = 3;
pulseType = 'wide';
% get current DWI protocol for simulation
bval_sorted     = [0.05, 0.3, 0.8, 1.5, 2.3, 3.5, 4.8, 6.5];
BDELTA_sorted   = [13,    13,  13,  13,  13,  13,  13,  13]; %ms
ldelta_sorted   = [6,    6,  6,  6,  6,  6,  6,  6]; %ms

% Parameter raneg for forward simulation
Rs_range    = [5, 10];
f_range     = [0.1, 0.8];
Da_range    = [1.5, 3];
De_range    = [0.5, 1.5];
fs_range    = [0.1, 0.8];

% generate ground truth
Rs_GT   = single(rand(1,Nsample) * diff(Rs_range) + min(Rs_range));
Da_GT   = single(rand(1,Nsample) * diff(Da_range)  + min(Da_range));
f_GT    = single(rand(1,Nsample) * diff(f_range)  + min(f_range));
De_GT   = single(rand(1,Nsample) * diff(De_range)  + min(De_range));
fs_GT   = single(rand(1,Nsample) * diff(fs_range)  + min(fs_range));

pars        = [];
pars.f      = f_GT;
pars.Da     = Da_GT;
pars.De     = De_GT;
pars.Rs     = Rs_GT;
pars.fs     = fs_GT;
objGPU      = gpuSANDI(bval_sorted, ldelta_sorted, BDELTA_sorted, Ds);
s           = objGPU.FWD(pars,pulseType);

% Let's assume Gaussian noise for simplicity
noiseLv = 1/SNR;
y       = s + randn(size(s)) .* noiseLv;
y       = permute(y,[3 2 4 1]);
mask    = ones(size(y,1:3)) > 0;

%% askadam estimation
rng(seed); gpurng(seed);

fitting.solver              = 'askadam';
fitting                     = objGPU.check_set_default(fitting);
fitting.start               = 'likelihood'; 
extraData                   = [];

out_adam                    = objGPU.estimate(y, mask, fitting, extraData);

% make some plots
% get initial starting point based on likelihood method for scatter plots
rng(seed); gpurng(seed);
fitting.iteration   = 0;
objGPU              = gpuSANDI(bval_sorted, ldelta_sorted, BDELTA_sorted, Ds);
pars0               = objGPU.estimate(y, mask, fitting, []); 

% plot result
field = fieldnames(pars);
figure;tiledlayout(2,numel(field)+1,"TileSpacing","compact");
for k = 1:numel(field)
    nexttile;
    scatter(pars.(field{k}),pars0.final.(field{k}),5,'filled','MarkerFaceAlpha',.4);hold on
    scatter(pars.(field{k}),out_adam.final.(field{k}),5,'filled','MarkerFaceAlpha',.4);
    h = refline(1);
    h.Color = 'k';
    title(field{k});
    xlabel('GT');ylabel('Fitted');
    drawnow
end

nexttile([1 numel(field)+1]);plot(pars0.final.resloss,'x');hold on;plot(out_adam.final.resloss,'o');title('Loss on each sample')
legend('Start','askadam.m estimation');xlabel('Sample');ylabel('loss');

%% MCMC estimation
% reset class object for MCMC
objGPU      = gpuSANDI(bval_sorted, ldelta_sorted, BDELTA_sorted, Ds);

fitting                     = [];
fitting.solver              = 'mcmc';
fitting                     = objGPU.check_set_default(fitting);
fitting.start               = 'likelihood';
fitting.algorithm           = 'ensemble';
fitting.Nwalker             = 30;
fitting.StepSize            = 2;
fitting.iteration           = 1e4;
fitting.thinning            = 10;        % Sample every 10 iteration
fitting.metric              = {'median','iqr'};
extraData                   = [];

out_mcmc   = objGPU.estimate(y, mask, fitting, extraData);

% plot result
figure;
field = fieldnames(pars);
tiledlayout(1,numel(field));
for k = 1:numel(field)
    nexttile;
    scatter(pars.(field{k}),out_mcmc.median.(field{k}),5,'filled','MarkerFaceAlpha',.4);
    h = refline(1);
    h.Color = 'k';
    title(field{k});
    xlabel('GT');ylabel('mcmc.m estimation');
    drawnow
end
