addpath(genpath('../../gacelle/'));
clear;
%% Simulate data

% for reproducibility
seed        = 23439; rng(seed); gpurng(seed);
Nsample     = 1e3;  % #voxel
SNR         = 40;  % at b0

% get current DWI protocol for simulation
bval_sorted     = [ 0,0.5, 1, 2, 6, 0, 0.5, 1, 2, 6];
te_sorted       = [54, 54,54,54,54,92,  92,92,92,92] * 1e-3;

% Parameter raneg for forward simulation
f_range    = [0.1, 0.8];
D_range    = [1.5, 3];
R2a_range  = [7,   14];
R2e_range  = [15,  50];

% generate ground truth
D_GT        = single(rand(1,Nsample) * diff(D_range)   + min(D_range));
f_GT        = single(rand(1,Nsample) * diff(f_range)   + min(f_range));
R2a_GT      = single(rand(1,Nsample) * diff(R2a_range) + min(R2a_range));
R2e_GT      = single(rand(1,Nsample) * diff(R2e_range) + min(R2e_range));

pars        = [];
pars.f      = f_GT;
pars.D      = D_GT;
pars.R2a    = R2a_GT;
pars.R2e    = R2e_GT;
objGPU      = gpumcmicro(bval_sorted,te_sorted);
s           = objGPU.FWD(pars);

% Let's assume Gaussian noise for simplicity
noiseLv = 1/SNR;
y       = s + randn(size(s)) .* noiseLv;
y       = permute(y,[3 2 4 1]);
mask    = ones(size(y,1:3)) > 0;

%% askadam estimation
rng(seed); gpurng(seed);

fitting             = [];
fitting.solver      = 'askadam';
fitting             = objGPU.check_set_default(fitting);
out                 = objGPU.estimate(y, mask, fitting);

%% make some plots
rng(seed); gpurng(seed);

% get initial starting point based on likelihood method for scatter plots
fitting.iteration   = 0;
pars0               = objGPU.estimate(y, mask, fitting); 

% get FWD signal based on fitted result
shat = objGPU.FWD(out.final);

%% plot result
field = fieldnames(pars);
figure;tiledlayout(2,numel(field),"TileSpacing","compact");
for k = 1:numel(field)
    nexttile;
    scatter(pars.(field{k}),pars0.final.(field{k}),5,'filled','MarkerFaceAlpha',.4);hold on
    scatter(pars.(field{k}),out.final.(field{k}),5,'filled','MarkerFaceAlpha',.4);
    h = refline(1);
    h.Color = 'k';
    title(field{k});
    xlabel('GT');ylabel('Fitted');
end
xlabel('GT');ylabel('Fitted');

nexttile([1 numel(field)]);plot(pars0.final.resloss,'x');hold on;plot(out.final.resloss,'o');title('Loss on each sample')
legend('Start','Fitted');xlabel('Sample');ylabel('loss');
