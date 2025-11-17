addpath(genpath('/autofs/space/linen_001/users/kwokshing/tools/gacelle/'));
clear;
%% Simulate data

% for reproducibility
seed        = 23439; rng(seed); gpurng(seed);
Nsample     = 1e3;  % #voxel
SNR         = 40;  % at b0

% get current DWI protocol for simulation
bval_sorted     = [0, 0.5, 1, 2];

% Parameter raneg for forward simulation
f_range    = [0.1, 0.8];
D_range    = [1.5, 3];

% generate ground truth
D_GT        = single(rand(1,Nsample) * diff(D_range)  + min(D_range));
f_GT        = single(rand(1,Nsample) * diff(f_range)  + min(f_range));

pars        = [];
pars.f      = f_GT;
pars.D      = D_GT;
objGPU      = gpumcmicro(bval_sorted);
s           = objGPU.FWD(pars);

% Let's assume Gaussian noise for simplicity
noiseLv = 1/SNR;
y       = s + randn(size(s)) .* noiseLv;
y       = gather(permute(y,[3 2 4 1]));
mask    = ones(size(y,1:3)) > 0;

%% askadam estimation
rng(seed); gpurng(seed);

fitting             = [];
fitting.solver      = 'mcmc';
fitting.algorithm   = 'ensemble';
fitting.Nwalker     = 50;
fitting.StepSize    = 2;
fitting.iteration   = 1e4;
fitting.thinning    = 10;        % Sample every 10 iteration
fitting.metric      = {'median','iqr'};
fitting.burnin      = 0.1;       % 10% burn-in
fitting.start       = 'likelihood';
extraData           = [];

objGPU              = gpumcmicro(bval_sorted);
out                 = objGPU.estimate(y, mask, fitting);

%% make some plots
rng(seed); gpurng(seed);

% get initial starting point based on likelihood method for scatter plots
objGPU              = gpumcmicro(bval_sorted);
fitting.iteration   = 0;
fitting.ub          = objGPU.ub;
fitting.lb          = objGPU.lb;
pars0               = objGPU.determine_x0(y,mask,fitting) ;
% pars0               = objGPU.estimate(y, mask, fitting); 

% get FWD signal based on fitted result
objGPU              = gpumcmicro(bval_sorted);
shat                = objGPU.FWD(out.median);

%% plot result
field = fieldnames(pars);
figure;tiledlayout(2,numel(field),"TileSpacing","compact");
for k = 1:numel(field)
    nexttile;
    scatter(pars.(field{k}),pars0.(field{k}),5,'filled','MarkerFaceAlpha',.4);hold on
    scatter(pars.(field{k}),out.median.(field{k}),5,'filled','MarkerFaceAlpha',.4);
    h = refline(1);
    h.Color = 'k';
    title(field{k});
    xlabel('GT');ylabel('Fitted');
end
xlabel('GT');ylabel('Fitted');

% nexttile([1 numel(field)]);plot(pars0.final.resloss,'x');hold on;plot(out.final.resloss,'o');title('Loss on each sample')
% legend('Start','Fitted');xlabel('Sample');ylabel('loss');

% %% plot residual
% figure;
% nexttile;h1 = plot(squeeze(y).','bo');hold on;h2 = plot(squeeze(shat),':r');h3 = plot(squeeze(s),'k');
% legend([h1(1),h2(1),h3(1)],'Noisy data','Fitted','Noiseless');
% nexttile;plot(out.final.residual.','--o');title('Fitting residuals')
