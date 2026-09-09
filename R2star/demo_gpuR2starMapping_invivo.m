%% demo_gpuAxCaliberSMTmcmc_RealData.m
%
% This demo provides several examples on the ulitisation of gpuJointR1R2starMapping.m 
% for parameter estimation with in vivo data
% 
% Kwok-Shing Chan 
% kchan2@mgh.harvard.edu
%
% Date created: 24 June 2026 
%
%% add paths
addpath('../../gacelle/');addpath_gacelle;
clear;

%% I/O: Load data
gre_invivo_dir = fullfile('~/Downloads','zenodo22666992'); % this is where the data locates, feel free to update this path
check_gre_invivo_demo_data; % check if the demo data exists, if not then download it to gre_invivo_dir

subj_label = 'sub-003';
sess_label = 'ses-mri01';

bids_dir            = fullfile(gre_invivo_dir,'bids');
derivatives_dir     = fullfile(bids_dir,'derivatives');
preproc_dir         = fullfile(derivatives_dir,'preprocessed');

%% Subject info and directories

file_list = dir(fullfile(preproc_dir,subj_label,sess_label,'anat','*_sepia_header.mat'));
flip_angle = zeros(1,numel(file_list));
for kfile = 1:numel(file_list)
    load(fullfile(file_list(kfile).folder,file_list(kfile).name),'FA');
    flip_angle(kfile) = FA;
end
flip_angle = sort(flip_angle,'ascend');

%% load data
counter         = 0;
img             = [];
sepia_header    = [];
unwrappedPhase  = [];
totalField      = [];
fa              = zeros(1,length(flip_angle));
for kfa = 1%:length(flip_angle) % get one multi-echo data only
    counter = counter + 1;
    
    FAcurr          = sprintf('%d', flip_angle(kfa));

     % general GRE basename
    acq_label   = strcat('acq-',['TR50NTE15FA' FAcurr]);
    prefix      = strcat(subj_label,'_',sess_label,'_',acq_label);

    % magnitude nifti image filename
    magn_fn         = dir(fullfile(preproc_dir,subj_label,sess_label,'anat',strcat(prefix,'_*part-mag*_MEGRE_space-withinGRE.nii.gz*')));
    sepia_header_fn = dir(fullfile(preproc_dir,subj_label,sess_label,'anat',strcat(prefix,'_*MEGRE_sepia_header.mat')));

    nii                 = niftiread(fullfile(magn_fn.folder, magn_fn.name));
    img                 = cat(5,img,nii);
    sepia_header{kfa}   = load(fullfile(sepia_header_fn.folder, sepia_header_fn.name));

    fa(kfa)  = sepia_header{kfa}.FA;
    tr      = sepia_header{kfa}.TR;

end
te = sepia_header{end}.TE;

mask_fn         = dir(fullfile(preproc_dir,subj_label,sess_label,'anat',strcat(subj_label,'*brain_mask*.nii*')));
mask_filename   = fullfile(mask_fn.folder, mask_fn.name);
mask            = niftiread(mask_filename);

%% Prepare data fot batch processing
objGPU              = gpuR2starMapping(te);

fitting             = [];
fitting.solver      = 'askadam';
fitting             = objGPU.check_set_default(fitting);
fitting.start       = 'default';

out_adam            = objGPU.estimate(img, mask, fitting);

%% DEMO#2: mcmc estimation, uniform weights
% reset class object
objGPU              = gpuR2starMapping(te);

fitting             = [];
fitting.solver      = 'mcmc';
fitting             = objGPU.check_set_default(fitting);

out_mh              = objGPU.estimate(img, mask, fitting);
