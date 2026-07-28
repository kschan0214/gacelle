%% add paths
addpath(genpath('../../gacelle')); % this is the path to 'gacelle' package
clear;

%% I/O: Load data
check_gre_invivo_demo_data; % check if the demo data exists

subj_label = 'sub-003';
sess_label = 'ses-mri01';

bids_dir            = fullfile(gre_invivo_dir,'bids');
derivatives_dir     = fullfile(bids_dir,'derivatives');
preproc_dir         = fullfile(derivatives_dir,'preprocessed');
sepia_dir           = fullfile(derivatives_dir,'sepia');

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

%% DEMO #1: Linear PDF
weights = sum(abs(img).^2,4);
weights = weights ./max(weights(:));

objGPU = gpuPDF(sepia_header.voxelSize, sepia_header.B0, sepia_header.B0_dir,sepia_header.delta_TE);

fitting                     = [];
fitting.tol                 = 1e-6;
fitting.initialLearnRate    = 0.01;
fitting.decayRate           = 0.001;
fitting.convergenceValue    = 1e-5;

extraData                   = [];
extraData.weights           = weights;

out = objGPU.estimate(totalField,mask,extraData,fitting);

