
addpath(genpath('/autofs/space/linen_001/users/kwokshing/tools/askadam/'))

FA = [20];

mag = [];
pha = [];

for kfa = 1:numel(FA)

    work_dir = '/autofs/space/symphony_002/users/kwokshing/project/askadam/external_data/TestRetestMCRMWI/bids/derivatives/ANTs/sub-002/ses-mri01/gre_within_protocol_space';

    mag = cat(4,mag, niftiread(fullfile(work_dir,['sub-002_ses-mri01_acq-TR38NTE12FA' num2str(FA(kfa)) '_run-1_part-mag_MEGRE_space-withinGRE.nii.gz'])));
    pha = cat(4,pha, niftiread(fullfile(work_dir,['sub-002_ses-mri01_acq-TR38NTE12FA' num2str(FA(kfa)) '_run-1_part-phase_MEGRE_space-withinGRE.nii.gz'])));

end
img = mag .* exp(1i*pha);
mask = min(mag > 0,[],4);

load('/autofs/space/symphony_002/users/kwokshing/project/askadam/external_data/TestRetestMCRMWI/bids/derivatives/SEPIA/sub-002/ses-mri01/TR38NTE12/sub-002_ses-mri01_acq-TR38NTE12FA20_run-1_MEGRE_sepia_header.mat')

totalField  = niftiread(fullfile('/autofs/space/symphony_002/users/kwokshing/project/askadam/external_data/TestRetestMCRMWI/bids/derivatives/SEPIA/sub-002/ses-mri01/TR38NTE12/FA20',['sub-002_ses-mri01_acq-TR38NTE12FA20_run-1_MEGRE_space-withinGRE_EchoCombine-OW_total-field.nii.gz']));
pini        = angle(exp(1i*pha(:,:,:,1)) ./ exp(1i* 2*pi*totalField .* permute(TE(1),[2 3 4 1])));

iwf = niftiread('/autofs/space/symphony_002/users/kwokshing/project/askadam/external_data/TestRetestMCRMWI/bids/derivatives/ANTs/sub-001/ses-mri01/dwi_2_gre_ses-mri01_space/sub-001_ses-mri01_run-1_dwi_denoised-patch2self_degibbs_eddy_smt-mcmicro_intra_space-withinGRE-TR38NTE12_ses-mri01.nii.gz');
ff(:,:,:,1) = niftiread('/autofs/space/symphony_002/users/kwokshing/project/askadam/external_data/TestRetestMCRMWI/bids/derivatives/ANTs/sub-001/ses-mri01/dwi_2_gre_ses-mri01_space/mean_f1samples_space-withinGRE-TR38NTE12_ses-mri01.nii.gz');
ff(:,:,:,2) = niftiread('/autofs/space/symphony_002/users/kwokshing/project/askadam/external_data/TestRetestMCRMWI/bids/derivatives/ANTs/sub-001/ses-mri01/dwi_2_gre_ses-mri01_space/mean_f2samples_space-withinGRE-TR38NTE12_ses-mri01.nii.gz');
ff(:,:,:,3) = niftiread('/autofs/space/symphony_002/users/kwokshing/project/askadam/external_data/TestRetestMCRMWI/bids/derivatives/ANTs/sub-001/ses-mri01/dwi_2_gre_ses-mri01_space/mean_f3samples_space-withinGRE-TR38NTE12_ses-mri01.nii.gz');

theta(:,:,:,1) = gpuGREMWI.AngleBetweenV1MapAndB0(niftiread('/autofs/space/symphony_002/users/kwokshing/project/askadam/external_data/TestRetestMCRMWI/bids/derivatives/ANTs/sub-001/ses-mri01/dwi_2_gre_ses-mri01_space/dyads1_space-withinGRE-TR38NTE12_ses-mri01.nii.gz'),B0_dir);
theta(:,:,:,2) = gpuGREMWI.AngleBetweenV1MapAndB0(niftiread('/autofs/space/symphony_002/users/kwokshing/project/askadam/external_data/TestRetestMCRMWI/bids/derivatives/ANTs/sub-001/ses-mri01/dwi_2_gre_ses-mri01_space/dyads2_space-withinGRE-TR38NTE12_ses-mri01.nii.gz'),B0_dir);
theta(:,:,:,3) = gpuGREMWI.AngleBetweenV1MapAndB0(niftiread('/autofs/space/symphony_002/users/kwokshing/project/askadam/external_data/TestRetestMCRMWI/bids/derivatives/ANTs/sub-001/ses-mri01/dwi_2_gre_ses-mri01_space/dyads3_space-withinGRE-TR38NTE12_ses-mri01.nii.gz'),B0_dir);


%%
kappa_mw                = 0.36; % Jung, NI., myelin water density
kappa_iew               = 0.86; % Jung, NI., intra-/extra-axonal water density
fixed_params.B0     	= 3;    % field strength, in tesla
fixed_params.rho_mw    	= kappa_mw/kappa_iew; % relative myelin water density
fixed_params.E      	= 0.02; % exchange effect in signal phase, in ppm
fixed_params.x_i      	= -0.1; % myelin isotropic susceptibility, in ppm
fixed_params.x_a      	= -0.1; % myelin anisotropic susceptibility, in ppm
fixed_params.B0dir      = B0_dir;

%%

objGPU                     = gpuGREMWI(TE,fixed_params);

fitting = [];
fitting.Nepoch              = 4000;
fitting.initialLearnRate    = 0.001;
fitting.decayRate           = 0;
fitting.convergenceValue    = 1e-8;
fitting.tol                 = 1e-3;
fitting.isdisplay           = 1;

fitting.DIMWI.isFitIWF      = 1;
fitting.DIMWI.isFitFreqMW   = 1;
fitting.DIMWI.isFitFreqIW   = 1;
fitting.DIMWI.isFitR2sEW    = 1;

extraData = [];
extraData.freqBKG   = totalField / (gpuGREMWI.gyro*fixed_params.B0); % in ppm
extraData.pini      = pini;
extraData.IWF       = iwf;
extraData.theta     = theta(:,:,:,:);
extraData.ff        = ff(:,:,:,:)./sum(ff,4);

[out]    = objGPU.estimate(img, mask, extraData, fitting);

%%
fitting.isComplex = 1;
[Sreal, Simag] = objGPU.FWD(out_askadam.final, fitting, extraData);
S = Sreal + 1i*Simag;
