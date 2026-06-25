
addpath(genpath('../../gacelle/'));
clear;

%% Simulate data

% for reproducibility
seed        = 23439; rng(seed);
Nsample     = 1e3;  % #voxel
SNR         = 150;

Nt  = 15; 
TE1 = 1.5e-3;
tr  = 45e-3;
t   = linspace(TE1, tr-3e-3, Nt);
B0  = 3;

% Parameter raneg for forward simulation
M0_range        = [0.5, 2];
MWF_range       = [1e-8, 0.25];
IWF_range       = [0.2, 0.8];
R2sMW_range     = [75, 150];
R2sIW_range     = [8, 65];
R2sp_range      = [0, 10];
freqMW_range    = [0, 20]/ B0/ gpuGREMWI.gyro;
freqIW_range    = [-10, 0]/ B0/ gpuGREMWI.gyro;
dfreqBKG_range  = [-0.05 0.05];
dpini           = [-1,1];

% generate ground truth
M0_GT       = single(rand(1,Nsample) * diff(M0_range) + min(M0_range) );
MWF_GT      = single(rand(1,Nsample) * diff(MWF_range) + min(MWF_range) );
IWF_GT      = single(rand(1,Nsample) * diff(IWF_range) + min(IWF_range) );
R2sMW_GT    = single(rand(1,Nsample) * diff(R2sMW_range) + min(R2sMW_range) );
R2sIW_GT    = single(rand(1,Nsample) * diff(R2sIW_range) + min(R2sIW_range) );
R2sEW_GT    = R2sIW_GT + single(rand(1,Nsample) * diff(R2sp_range) + min(R2sp_range) );
freqMW_GT   = single(rand(1,Nsample) * diff(freqMW_range) + min(freqMW_range) );
freqIW_GT   = single(rand(1,Nsample) * diff(freqIW_range) + min(freqIW_range) );
dfreqBKG_GT = single(rand(1,Nsample) * diff(dfreqBKG_range) + min(dfreqBKG_range) );
pini_GT     = single(rand(1,Nsample) * diff(dpini) + min(dpini) );

% forward signal simulation
pars        = [];
pars.S0     = M0_GT;
pars.MWF    = MWF_GT;
pars.IWF    = IWF_GT;
pars.R2sMW  = R2sMW_GT;
pars.R2sIW  = min(R2sIW_GT,50);
pars.R2sEW  = min(R2sEW_GT,50);
pars.freqMW = freqMW_GT;
pars.freqIW = freqIW_GT;
pars.dfreqBKG= dfreqBKG_GT;
pars.dpini    = pini_GT;

fitting.DIMWI.isFitIWF      = true;
fitting.DIMWI.isFitFreqMW   = true;
fitting.DIMWI.isFitFreqIW   = true;
fitting.DIMWI.isFitR2sEW    = true;
fitting.isComplex = true;

extraData.freqBKG   = zeros(size(M0_GT,1:3));
extraData.pini      = zeros(size(M0_GT,1:3));
extraData.ff        = ones([size(M0_GT,1:3)]); %extraData.ff = extraData.ff ./ sum(extraData.ff,ndims(extraData.ff ));
extraData.theta     = zeros([size(M0_GT,1:3)]);

objGPU  = gpuGREMWI(t);
s       = ((objGPU.FWD(pars, fitting, extraData))).';
s       = permute(reshape(s,[Nsample,Nt,2]),[1 4 5 2 3]);
mask    = ones(size(s,1:3),'logical');

% Let's assume Gaussian noise for simplicity
noiseLv = (M0_GT.')./SNR;
y       = s + randn(size(s)) .* noiseLv;
y       = y(:,:,:,:,1) + 1i*y(:,:,:,:,2);

%% DEMO#1: askadam estimation
fitting                     = [];
fitting.solver              = 'askadam';
fitting                     = objGPU.check_set_default(fitting,y);
fitting.initialLearnRate    = 0.01;
fitting.convergenceValue    = 1e-5;
fitting.start               = 'prior';   

objGPU      = gpuGREMWI(t);
out_adam    = objGPU.estimate(y, mask, extraData, fitting);

% starting position
pars0   = objGPU.estimate_prior(y);

% plot result
figure(98);
field = fieldnames(pars);
tiledlayout(1,numel(field));
for k = 1:numel(field)
    nexttile;
    scatter(pars.(field{k}),pars0.(field{k}),5,'filled','MarkerFaceAlpha',.4);hold on
    scatter(pars.(field{k}),out_adam.final.(field{k}),5,'filled','MarkerFaceAlpha',.4);
    h = refline(1);
    h.Color = 'k';
    title(field{k});
    xlabel('GT');ylabel('Fitted');
end

legend('Start','askadam.m estimation','Location','northwest');

%% DEMO#2: mcmc estimation
fitting                     = [];
fitting.solver              = 'mcmc';
fitting                     = objGPU.check_set_default(fitting,y);
fitting.algorithm           = 'ensemble';
fitting.Nwalker             = 30;
fitting.StepSize            = 2;
fitting.iteration           = 1e4;
fitting.thinning            = 10;        % Sample every 10 iteration
fitting.metric              = {'median','iqr'};
fitting.burnin              = 0.1;       % 10% burn-in
fitting.start               = 'prior';

objGPU          = gpuGREMWI(t);
out_ensemble    = objGPU.estimate(y, mask, extraData, fitting);

% plot result
figure(99);
field = fieldnames(pars);
tiledlayout(1,numel(field));
for k = 1:numel(field)
    nexttile;
    scatter(pars.(field{k}),out_ensemble.median.(field{k}),5,'filled','MarkerFaceAlpha',.4);
    h = refline(1);
    h.Color = 'k';
    title(field{k});
    xlabel('GT');ylabel('Fitted');
end

legend('mcmc.m estimation','Location','northwest');