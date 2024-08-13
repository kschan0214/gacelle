
addpath(genpath('/autofs/space/linen_001/users/kwokshing/tools/askadam/'))

FA = [5,10,20,50,70];

mag = [];
pha = [];
totalField = [];

for kfa = 1:numel(FA)

    work_dir = '/autofs/space/symphony_002/users/kwokshing/project/askadam/gpudimwi/bids/ANTs/sub-002/ses-mri01/gre_within_protocol_space';

    mag = cat(5,mag, niftiread(fullfile(work_dir,['sub-002_ses-mri01_acq-TR38NTE12FA' num2str(FA(kfa)) '_run-1_part-mag_MEGRE_space-withinGRE.nii.gz'])));
    pha = cat(5,pha, niftiread(fullfile(work_dir,['sub-002_ses-mri01_acq-TR38NTE12FA' num2str(FA(kfa)) '_run-1_part-phase_MEGRE_space-withinGRE.nii.gz'])));

    work_dir = ['/autofs/space/symphony_002/users/kwokshing/project/askadam/gpudimwi/bids/SEPIA/sub-002/ses-mri01/TR38NTE12/FA' num2str(FA(kfa))];

    totalField = cat(5,totalField, niftiread(fullfile(work_dir,['sub-002_ses-mri01_acq-TR38NTE12FA' num2str(FA(kfa)) '_run-1_MEGRE_space-withinGRE_EchoCombine-OW_total-field.nii.gz'])));

end
img = mag .* exp(1i*pha);
mask = min(min(mag > 0,[],4),[],5);

load('/autofs/space/symphony_002/users/kwokshing/project/askadam/gpudimwi/bids/SEPIA/sub-002/ses-mri01/TR38NTE12/sub-002_ses-mri01_acq-TR38NTE12FA20_run-1_MEGRE_sepia_header.mat')

% don't use high flip angle as the phase get weird
pini = squeeze(exp(1i*pha(:,:,:,1,:)) ./ exp(1i* 2*pi*totalField .* permute(TE(1),[2 3 4 1])));
pini = angle(mean(pini(:,:,:,1:3),4));

b1 = niftiread(fullfile('/autofs/space/symphony_002/users/kwokshing/project/askadam/gpudimwi/bids/ANTs/sub-002/ses-mri01/b1_2_gre_space/sub-002_ses-mri01_acq-famp_run-1_TB1TFL_space-withinGRE-TR38NTE12.nii.gz'))/10/80;

iwf = niftiread('/autofs/space/symphony_002/users/kwokshing/project/askadam/gpudimwi/bids/ANTs/sub-001/ses-mri01/dwi_2_gre_ses-mri01_space/sub-001_ses-mri01_run-1_dwi_denoised-patch2self_degibbs_eddy_smt-mcmicro_intra_space-withinGRE-TR38NTE12_ses-mri01.nii.gz');
ff(:,:,:,1) = niftiread('/autofs/space/symphony_002/users/kwokshing/project/askadam/gpudimwi/bids/ANTs/sub-001/ses-mri01/dwi_2_gre_ses-mri01_space/mean_f1samples_space-withinGRE-TR38NTE12_ses-mri01.nii.gz');
ff(:,:,:,2) = niftiread('/autofs/space/symphony_002/users/kwokshing/project/askadam/gpudimwi/bids/ANTs/sub-001/ses-mri01/dwi_2_gre_ses-mri01_space/mean_f2samples_space-withinGRE-TR38NTE12_ses-mri01.nii.gz');
ff(:,:,:,3) = niftiread('/autofs/space/symphony_002/users/kwokshing/project/askadam/gpudimwi/bids/ANTs/sub-001/ses-mri01/dwi_2_gre_ses-mri01_space/mean_f3samples_space-withinGRE-TR38NTE12_ses-mri01.nii.gz');

theta(:,:,:,1) = gpuGREMWI.AngleBetweenV1MapAndB0(niftiread('/autofs/space/symphony_002/users/kwokshing/project/askadam/gpudimwi/bids/ANTs/sub-001/ses-mri01/dwi_2_gre_ses-mri01_space/dyads1_space-withinGRE-TR38NTE12_ses-mri01.nii.gz'),B0_dir);
theta(:,:,:,2) = gpuGREMWI.AngleBetweenV1MapAndB0(niftiread('/autofs/space/symphony_002/users/kwokshing/project/askadam/gpudimwi/bids/ANTs/sub-001/ses-mri01/dwi_2_gre_ses-mri01_space/dyads2_space-withinGRE-TR38NTE12_ses-mri01.nii.gz'),B0_dir);
theta(:,:,:,3) = gpuGREMWI.AngleBetweenV1MapAndB0(niftiread('/autofs/space/symphony_002/users/kwokshing/project/askadam/gpudimwi/bids/ANTs/sub-001/ses-mri01/dwi_2_gre_ses-mri01_space/dyads3_space-withinGRE-TR38NTE12_ses-mri01.nii.gz'),B0_dir);

%%
kappa_mw                = 0.36; % Jung, NI., myelin water density
kappa_iew               = 0.86; % Jung, NI., intra-/extra-axonal water density
fixed_params.B0     	= 3;    % field strength, in tesla
fixed_params.rho_mw    	= kappa_mw/kappa_iew; % relative myelin water density
fixed_params.E      	= 0.02; % exchange effect in signal phase, in ppm
fixed_params.x_i      	= -0.1; % myelin isotropic susceptibility, in ppm
fixed_params.x_a      	= -0.1; % myelin anisotropic susceptibility, in ppm
fixed_params.B0dir      = B0_dir;
fixed_params.t1_mw      = 234e-3;

%%
FA = [5,10,20,50,70];
objGPU                     = gpuMCRMWI(TE,TR,FA,fixed_params);

fitting = [];
fitting.Nepoch              = 4000;
fitting.initialLearnRate    = 0.01;
fitting.decayRate           = 0;
fitting.convergenceValue    = 1e-8;
fitting.tol                 = 1e-4;
fitting.display             = false;

fitting.DIMWI.isFitIWF      = 0;
fitting.DIMWI.isFitFreqMW   = 0;
fitting.DIMWI.isFitFreqIW   = 0;
fitting.DIMWI.isFitR2sEW    = 0;
fitting.isFitExchange       = 1;
fitting.isEPG               = 1;

extraData = [];
extraData.freqBKG   = single(squeeze(totalField) / (gpuGREMWI.gyro*fixed_params.B0)); % in ppm
extraData.pini      = single(pini);
extraData.IWF       = single(iwf);
extraData.theta     = single(theta);
extraData.ff        = single(ff./sum(ff,4));
extraData.b1        = single(b1);


[out_askadam_mcr]    = objGPU.estimate(img, mask, extraData, fitting);


%%
te = linspace(0,50e-3,15);
tr = 55e-3;
fa = [5,10,20,30,40,50,70];
fixed_params_tmp = fixed_params;
fixed_params_tmp.rho_mw = 0.42;
objGPU                     = gpuMCRMWI(te,tr,fa,fixed_params);

pars.S0     = 1;
pars.MWF    = 0.15;
pars.IWF    = 0.6/(1-0.15);
pars.R1IEW  = 1;
pars.kIEWM = 2;
pars.R2sMW = 1/10e-3;
pars.R2sIW = 1/64e-3;
pars.R2sEW = 1/48e-3;
pars.freqMW = 15/(objGPU.gyro*3);
pars.freqIW = -2/(objGPU.gyro*3);
pars.dfreqBKG = 0;
pars.dpini = 0;

extraData_tmp.b1 = 1;
extraData_tmp.freqBKG = [1:7]*10/(objGPU.gyro*3);
extraData_tmp.pini = -1;
extraData_tmp.ff = 1;
extraData_tmp.theta = 0;

fitting.isComplex = 1;
fitting.isEPG = 0;
fitting.isFitExchange       = 0;

ann_epgx_phase = load('MCRMWI_MLP_EPGX_leakyrelu_N2e6_phase_v2.mat','dlnet');
ann_epgx_phase.dlnet.alpha = 0.01;
ann_epgx_magn  = load('MCRMWI_MLP_EPGX_leakyrelu_N2e6_magn.mat','dlnet');
ann_epgx_magn.dlnet.alpha = 0.01;
[Sreal,Simag] = objGPU.FWD( pars, fitting, extraData_tmp,1, ann_epgx_phase.dlnet   ,ann_epgx_magn.dlnet);

S = reshape(Sreal,[15,7]) + 1i*reshape(Simag,[15,7]);