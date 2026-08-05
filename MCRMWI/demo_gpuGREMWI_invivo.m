%% add paths
addpath('../../gacelle'); addpath_gacelle; % this is the path to 'gacelle' package
clear;

%% I/O: Load data
check_gre_invivo_demo_data; % check if the demo data exists

subj_label = 'sub-003';
sess_label = 'ses-mri01';

bids_dir            = fullfile(gre_invivo_dir,'bids');
derivatives_dir     = fullfile(bids_dir,'derivatives');
preproc_dir         = fullfile(derivatives_dir,'preprocessed');
sepia_dir           = fullfile(derivatives_dir,'sepia');
fsl_dir             = fullfile(derivatives_dir,'fsl');
mcmicro_dir         = fullfile(derivatives_dir,'mcmicro');

%% Subject info and directories

file_list = dir(fullfile(preproc_dir,subj_label,sess_label,'anat','*_sepia_header.mat'));
flip_angle = zeros(1,numel(file_list));
for kfile = 1:numel(file_list)
    load(fullfile(file_list(kfile).folder,file_list(kfile).name),'FA');
    flip_angle(kfile) = FA;
end
flip_angle = sort(flip_angle,'ascend');

%% only work on FA20
kfa = 3;
FAcurr          = sprintf('%d', flip_angle(kfa));

% general GRE basename
acq_label   = strcat('acq-',['TR50NTE15FA' FAcurr]);
prefix      = strcat(subj_label,'_',sess_label,'_',acq_label);

% magnitude nifti image filename
magn_fn         = dir(fullfile(preproc_dir,subj_label,sess_label,'anat',strcat(prefix,'_*part-mag*_MEGRE_space-withinGRE.nii.gz*')));
phas_fn         = dir(fullfile(preproc_dir,subj_label,sess_label,'anat',strcat(prefix,'_*part-phase*_MEGRE_space-withinGRE.nii.gz*')));
sepia_header_fn = dir(fullfile(preproc_dir,subj_label,sess_label,'anat',strcat(prefix,'_*MEGRE_sepia_header.mat')));

img             = niftiread(fullfile(magn_fn.folder, magn_fn.name)) .*exp(1i*niftiread(fullfile(phas_fn.folder, phas_fn.name)));
sepia_header    = load(fullfile(sepia_header_fn.folder, sepia_header_fn.name));

te = sepia_header.TE;

mask_fn         = dir(fullfile(preproc_dir,subj_label,sess_label,'anat',strcat(subj_label,'*brain_mask*.nii*')));
mask_filename   = fullfile(mask_fn.folder, mask_fn.name);
mask            = niftiread(mask_filename);

% load other data
totalField_fn   = dir(fullfile(sepia_dir,subj_label,sess_label,'anat',strcat('FA',FAcurr),strcat(prefix,'_*fieldmap.nii.gz*')));
totalField      = niftiread(fullfile(totalField_fn.folder, totalField_fn.name)) ;
% DIMWI
iwf_fn          = dir(fullfile(mcmicro_dir,subj_label,sess_label,'dwi',strcat('*intra*.nii.gz*')));
iwf             = niftiread(fullfile(iwf_fn.folder, iwf_fn.name)) ;
fibreFrac = []; fibreOrient = [];
for k = 1:3
tmp_fn          = dir(fullfile(fsl_dir,subj_label,sess_label,'dwi',strcat('mean_f',num2str(k),'*.nii.gz*')));
fibreFrac       = cat(4,fibreFrac,niftiread(fullfile(tmp_fn.folder, tmp_fn.name)) );

tmp_fn          = dir(fullfile(fsl_dir,subj_label,sess_label,'dwi',strcat('dyads',num2str(k),'_space-withinGRE*.nii.gz*')));
fibreOrient     = cat(5,fibreOrient,niftiread(fullfile(tmp_fn.folder, tmp_fn.name)) );
end

% estimate initial phase
pini        = angle(img(:,:,:,1) ./ exp(1i* 2*pi*totalField .* permute(te(1),[2 3 4 1])));

%% fixed parameters
kappa_mw                = 0.36; % Jung, NI., myelin water density
kappa_iew               = 0.86; % Jung, NI., intra-/extra-axonal water density
fixed_params.B0     	= 3;    % field strength, in tesla
fixed_params.rho_mw    	= kappa_mw/kappa_iew; % relative myelin water density
fixed_params.E      	= 0.02; % exchange effect in signal phase, in ppm
fixed_params.x_i      	= -0.1; % myelin isotropic susceptibility, in ppm
fixed_params.x_a      	= -0.1; % myelin anisotropic susceptibility, in ppm
fixed_params.B0dir      = sepia_header.B0_dir;

%% DEMO #1: conventional GRE-MWI
objGPU                  = gpuGREMWI(te,fixed_params);

fitting                     = [];
fitting.solver              = 'askadam';
fitting                     = objGPU.check_set_default(fitting,img);
fitting.start               = 'prior';   
fitting.initialLearnRate    = 0.01;
fitting.convergenceValue    = 1e-5;
fitting.weightPower         = 0.5;


fitting.DIMWI.isFitIWF      = 1;
fitting.DIMWI.isFitFreqMW   = 1;
fitting.DIMWI.isFitFreqIW   = 1;
fitting.DIMWI.isFitR2sEW    = 1;

extraData           = [];
extraData.freqBKG   = totalField / (gpuGREMWI.gyro*fixed_params.B0); % in ppm
extraData.pini      = pini;

out_GREMWI_adam    = objGPU.estimate(img, mask, extraData, fitting);

%% DEMO #2: DIMWI
objGPU                     = gpuGREMWI(te,fixed_params);

fitting                     = [];
fitting.solver              = 'askadam';
fitting                     = objGPU.check_set_default(fitting,img);
fitting.start               = 'prior';   
fitting.initialLearnRate    = 0.01;
fitting.convergenceValue    = 1e-5;
fitting.weightPower         = 0.5;

fitting.DIMWI.isFitIWF      = 0;
fitting.DIMWI.isFitFreqMW   = 0;
fitting.DIMWI.isFitFreqIW   = 0;
fitting.DIMWI.isFitR2sEW    = 0;

extraData = [];
extraData.freqBKG   = totalField / (gpuGREMWI.gyro*fixed_params.B0); % in ppm
extraData.pini      = pini;
extraData.IWF       = iwf;
extraData.fo        = fibreOrient;
extraData.ff        = fibreFrac(:,:,:,:)./sum(fibreFrac,4);

out_DIMWI_adam    = objGPU.estimate(img, mask, extraData, fitting);

%% DEMO #3: conventional GRE-MWI with MCMC
objGPU                      = gpuGREMWI(te,fixed_params);

fitting.solver              = 'mcmc';
fitting                     = objGPU.check_set_default(fitting,img);
fitting.algorithm           = 'mh';
% fitting.Nwalker             = 20;
% fitting.StepSize            = 2;
fitting.iteration           = 1e6;
fitting.thinning            = 100;        % Sample every 10 iteration
fitting.metric              = {'median','iqr'};
fitting.burnin              = 0.1;       % 10% burn-in
fitting.start               = 'prior';
fitting.weightPower         = 0.5;


fitting.DIMWI.isFitIWF      = 1;
fitting.DIMWI.isFitFreqMW   = 1;
fitting.DIMWI.isFitFreqIW   = 1;
fitting.DIMWI.isFitR2sEW    = 1;

extraData           = [];
extraData.freqBKG   = totalField / (gpuGREMWI.gyro*fixed_params.B0); % in ppm
extraData.pini      = pini;

out_GREMWI_mcmc    = objGPU.estimate(img, mask, extraData, fitting);
