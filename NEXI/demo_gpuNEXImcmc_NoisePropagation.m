addpath(genpath('/autofs/space/linen_001/users/kwokshing/tools/askadam/'));
clear;
%% Simulate data

% for reproducibility
seed        = 23439; rng(seed); gpurng(seed);
Nsample     = 1e3;  % #voxel
SNR         = 50;

% get current DWI protocol for simulation
bval_sorted     = [2.3, 3.5, 4.8, 6.5, 2.3, 3.5, 4.8, 6.5, 11.0, 2.3, 3.5, 4.8, 6.5, 11.0, 17.5];
BDELTA_sorted   = [13, 13, 13, 13, 21, 21, 21, 21, 21, 30, 30, 30, 30, 30, 30]; %ms

% Parameter raneg for forward simulation
ra_range    = [1/250, 0.3];
tex_range   = [2, 50];
fa_range    = [0.01, 0.8];
Da_range    = [1.5, 3];
De_range    = [0.5, 1.5];
p2_range    = [0.05, 0.5];
% noise_range     = [0.01 0.05];

% generate ground truth
tex_GT  = single(rand(1,Nsample) * diff(tex_range) + min(tex_range) );
Da_GT   = single(rand(1,Nsample) * diff(Da_range)  + min(Da_range));
fa_GT   = single(rand(1,Nsample) * diff(fa_range)  + min(fa_range));
De_GT   = single(rand(1,Nsample) * diff(De_range)  + min(De_range));
p2_GT   = single(rand(1,Nsample) * diff(p2_range)  + min(p2_range));
ra_GT   = (1-fa_GT) ./ tex_GT; 

lmax        = 2;
pars        = [];
pars.fa     = fa_GT;
pars.Da     = Da_GT;
pars.De     = De_GT;
pars.ra     = ra_GT;
pars.p2     = p2_GT;
objGPU      = gpuNEXImcmc(bval_sorted, BDELTA_sorted);
s           = gather(objGPU.FWD(pars, lmax));

% Let assume Gaussian noise to simplify everything
noiseLv = 1/SNR;
s       = s + randn(size(s)) .* noiseLv;
s       = permute(s,[2 3 4 1]);
mask    = ones(size(s,1:3));

%% MCMC estimation
fitting             = [];
fitting.algorithm   = 'GW';
fitting.Nwalker     = 50;
fitting.StepSize    = 2;
fitting.iteration   = 5e3;
fitting.thinning    = 20;        % Sample every 20 iteration
fitting.metric      = 'median';
fitting.burnin      = 0.1;       % 10% burn-in
fitting.lmax        = 2;     
fitting.start       = 'likelihood';     
extraData           = [];

objGPU  = gpuNEXImcmc(bval_sorted, BDELTA_sorted);
out     = objGPU.estimate(s, mask, extraData, fitting);

%% plot result
field = fieldnames(pars);
tiledlayout(1,numel(field)+1);
for k = 1:numel(field)
    nexttile;
    scatter(pars.(field{k}),out.median.(field{k}),5,'filled','MarkerFaceAlpha',.4);
    h = refline(1);
    h.Color = 'k';
    title(field{k});
    xlabel('GT');ylabel('Fitted');
end
nexttile;
scatter((1-pars.fa)./pars.ra,(1-out.median.fa)./out.median.ra,5,'filled','MarkerFaceAlpha',.4);
h = refline(1);
h.Color = 'k';
title('tex');
xlabel('GT');ylabel('Fitted');
