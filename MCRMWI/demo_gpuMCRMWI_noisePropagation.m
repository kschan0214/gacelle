
addpath(genpath('../../askadam/'));
clear;

%% Simulate data

% for reproducibility
seed        = 23439; rng(seed);
Nsample     = 1e4;  % #voxel
SNR         = 100;

Nt  = 15; 
TE1 = 1.5e-3;
tr  = 45e-3;
fa  = [5, 10, 20, 30, 40, 50, 70];
te  = linspace(TE1, tr-3e-3, Nt);

kappa_mw                = 0.36; % Jung, NI., myelin water density
kappa_iew               = 0.86; % Jung, NI., intra-/extra-axonal water density
fixed_params.B0     	= 3;    % field strength, in tesla
fixed_params.rho_mw    	= kappa_mw/kappa_iew; % relative myelin water density
fixed_params.E      	= 0.02; % exchange effect in signal phase, in ppm
fixed_params.x_i      	= -0.1; % myelin isotropic susceptibility, in ppm
fixed_params.x_a      	= -0.1; % myelin anisotropic susceptibility, in ppm
fixed_params.B0dir      = [0;0;1];
fixed_params.t1_mw      = 234e-3;
objGPU                  = gpuMCRMWI(te,tr,fa,fixed_params);

Nfa = numel(fa);

% Parameter raneg for forward simulation
M0_range        = [0.5, 2];
MWF_range       = [1e-8, 0.25];
IWF_range       = [0.2, 0.8];
R2sMW_range     = 1./[20e-3, 5e-3];
R2sIW_range     = 1./[120e-3, 60e-3];
dR2star_range   = [0, 20];
R1FW_range      = 1./[2.5, 0.8];
kFWM_range      = [1e-6, 2];
freqMW_range    = [0, 15] / (objGPU.B0*objGPU.gyro);
freqIW_range    = [-4, 0] / (objGPU.B0*objGPU.gyro);
dfreqBKG_range  = [-0.05 0.05];
dpini           = [-1,1];

% generate ground truth
M0_GT       = single(rand(1,Nsample) * diff(M0_range) + min(M0_range) );
MWF_GT      = single(rand(1,Nsample) * diff(MWF_range) + min(MWF_range) );
IWF_GT      = single(rand(1,Nsample) * diff(IWF_range) + min(IWF_range) );
R2sMW_GT    = single(rand(1,Nsample) * diff(R2sMW_range) + min(R2sMW_range) );
R2sIW_GT    = min(single(rand(1,Nsample) * diff(R2sIW_range) + min(R2sIW_range)),50);
R2sEW_GT    = min(R2sIW_GT + single(rand(1,Nsample) * diff(dR2star_range) + min(dR2star_range)),50);
freqMW_GT   = single(rand(1,Nsample) * diff(freqMW_range) + min(freqMW_range) );
freqIW_GT   = single(rand(1,Nsample) * diff(freqIW_range) + min(freqIW_range) );
dfreqBKG_GT = single(rand(1,Nsample,1,1,Nfa) * diff(dfreqBKG_range) + min(dfreqBKG_range) );
pini_GT     = single(rand(1,Nsample) * diff(dpini) + min(dpini) );
R1FW_GT     = single(rand(1,Nsample) * diff(R1FW_range) + min(R1FW_range));
kFWM_GT     = single(rand(1,Nsample) * diff(kFWM_range) + min(kFWM_range));

% forward signal simulation
pars        = [];
pars.S0     = M0_GT;
pars.MWF    = MWF_GT;
pars.IWF    = IWF_GT;
pars.R2sMW  = R2sMW_GT;
pars.R2sIW  = R2sIW_GT;
pars.R2sEW  = R2sEW_GT;
pars.freqMW = freqMW_GT;
pars.freqIW = freqIW_GT;
pars.R1IEW    = R1FW_GT;
pars.kIEWM    = kFWM_GT;
pars.dpini    = pini_GT;
pars.dfreqBKG= dfreqBKG_GT;

fitting.DIMWI.isFitIWF      = true;
fitting.DIMWI.isFitFreqMW   = true;
fitting.DIMWI.isFitFreqIW   = true;
fitting.DIMWI.isFitR2sEW    = true;
fitting.isFitExchange       = true;
fitting.isEPG               = true;
fitting.isComplex = true;

extraData.freqBKG   = zeros(size(M0_GT,1:3));
extraData.pini      = zeros(size(M0_GT,1:3));
extraData.ff        = ones([size(M0_GT,1:3)]); %extraData.ff = extraData.ff ./ sum(extraData.ff,ndims(extraData.ff ));
extraData.theta     = zeros([size(M0_GT,1:3)]);
extraData.b1        = ones([size(M0_GT,1:3)]);

dlnet_magn = load('/autofs/space/linen_001/users/kwokshing/tools/askadam/MCRMWI/EPGXgen_net/MCRMWI_MLP_EPGX_RFphase50_T1M234_magn.mat');
dlnet_phase = load('/autofs/space/linen_001/users/kwokshing/tools/askadam/MCRMWI/EPGXgen_net/MCRMWI_MLP_EPGX_RFphase50_T1M234_phase.mat');
dlnet_phase.dlnet.alpha = 0.01;
dlnet_magn.dlnet.alpha = 0.01;

mask        = ones(size(M0_GT,1:3),'logical');

objGPU     = gpuMCRMWI(te,tr,fa,fixed_params);
Sgpu_GT       = gather(extractdata(objGPU.FWD(pars, fitting, extraData,dlnet_phase.dlnet,dlnet_magn.dlnet)));
Sgpu_GT       = reshape(utils.reshape_AD2ND(Sgpu_GT,mask),[1 Nsample 1 Nt Nfa 2]); 
Sgpu_GT       = Sgpu_GT(:,:,:,:,:,1) + 1i*Sgpu_GT(:,:,:,:,:,2);

Smax = mean(M0_GT);
noise_sigma = Smax/SNR;
y = Sgpu_GT + noise_sigma*randn(size(Sgpu_GT)) + 1i*noise_sigma*randn(size(Sgpu_GT));

%% askadam estimation
fitting                     = [];
fitting.optimiser           = 'adam';
fitting.iteration           = 10000;
fitting.initialLearnRate    = 0.002;
fitting.decayRate           = 0.001;
fitting.convergenceValue    = 1e-8;
fitting.lossFunction        = 'l1';
fitting.tol                 = 1e-8;
fitting.isDisplay           = false;
fitting.start               = 'prior';   
fitting.patience            = 20;   
% extraData                   = [];

objGPU     = gpuMCRMWI(te,tr,fa,fixed_params);
out     = objGPU.estimate(y, mask, extraData, fitting);

%% starting position
objGPU     = gpuMCRMWI(te,tr,fa,fixed_params);
pars0   = objGPU.estimate_prior(y,mask,extraData);

%% plot result
figure(99);
field = fieldnames(pars);
tiledlayout(1,numel(field)-1);
for k = 1:numel(field)-1
    nexttile;
    scatter(pars.(field{k}),pars0.(field{k}),5,'filled','MarkerFaceAlpha',.4);hold on
    scatter(pars.(field{k}),out.final.(field{k}),5,'filled','MarkerFaceAlpha',.4);
    h = refline(1);
    h.Color = 'k';
    title(field{k});
    xlabel('GT');ylabel('Fitted');
end

legend('Start','Fitted','Location','northwest');