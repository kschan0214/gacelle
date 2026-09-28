%% demo_gpuJointR1R2starMapping_invivo_bayes.m
%
% This demo shows the Bayesian priors of the EXPERIMENTAL mcmc_bayes sampler with gpuJointR1R2starMapping.m
% on in vivo data:
%   Demo #1: BSP, a hierarchical (population) prior on the model parameters, learned from the
%            data together with the voxel parameters (Spinner et al. 2021, doi:10.1016/j.media.2021.102144)
%   Demo #2: BSP + spatial prior, two-stage empirical Bayes: stage 1 learns the population prior
%            (as Demo #1), stage 2 fixes it and adds a Markov random field (MRF) prior that couples
%            neighbouring voxels
% mcmc_bayes is selected with fitting.mcmcClass = 'mcmc_bayes'; see utils/mcmc_bayes.m for all options.
%
% Notes
% - Both priors couple all voxels, so the whole volume must be fitted in one GPU call. This demo
%   uses a slab of slices; set 'slab' to all slices if the GPU memory allows (otherwise
%   estimate() stops with the error gpuJointR1R2starMapping:singleSegment).
% - Use a new class object for each fit (fit() adapts the parameter list to the solver).
% - Check convergence before interpreting the maps: out.diagnostics.rhat.<param> (voxel chains,
%   aim for <= 1.01) and out.hyper.rhat (population prior; Demo #2: out.stage1.hyper.rhat).
%
% Kwok-Shing Chan
% kchan2@mgh.harvard.edu
%
% Date created: 28 September 2026
%
%% add paths
addpath('../../gacelle'); addpath_gacelle; % this is the path to 'gacelle' package
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

%% load data: all flip angles
img             = [];
fa              = zeros(1,length(flip_angle));
for kfa = 1:length(flip_angle)
    FAcurr          = sprintf('%d', flip_angle(kfa));
    acq_label       = strcat('acq-',['TR50NTE15FA' FAcurr]);
    prefix          = strcat(subj_label,'_',sess_label,'_',acq_label);

    magn_fn         = dir(fullfile(preproc_dir,subj_label,sess_label,'anat',strcat(prefix,'_*part-mag*_MEGRE_space-withinGRE.nii.gz*')));
    sepia_header_fn = dir(fullfile(preproc_dir,subj_label,sess_label,'anat',strcat(prefix,'_*MEGRE_sepia_header.mat')));
    img             = cat(5,img,niftiread(fullfile(magn_fn.folder, magn_fn.name)));
    sepia_header    = load(fullfile(sepia_header_fn.folder, sepia_header_fn.name));
    fa(kfa)         = sepia_header.FA;
    tr              = sepia_header.TR;
end
te = sepia_header.TE;

% B1 info
true_flip_angle_fn  = dir(fullfile(preproc_dir,subj_label,sess_label,'anat','*acq-famp*TB1TFL*space-withinGRE*.nii*'));
b1                  = niftiread( fullfile( true_flip_angle_fn.folder, true_flip_angle_fn.name)) / 10 / 80;

mask_fn         = dir(fullfile(preproc_dir,subj_label,sess_label,'anat',strcat(subj_label,'*brain_mask*.nii*')));
mask            = niftiread(fullfile(mask_fn.folder, mask_fn.name)) > 0;

%% slab of slices (see Notes)
slab            = round(size(mask,3)/2) + (-4:3);
img             = img(:,:,slab,:,:);
mask            = mask(:,:,slab);
extraData       = [];
extraData.b1    = b1(:,:,slab);

%% Bayesian MCMC settings shared by both demos
fitting                     = [];
fitting.solver              = 'mcmc';
fitting.mcmcClass           = 'mcmc_bayes';      % experimental Bayesian sampler
fitting.algorithm           = 'MH';
fitting.iteration           = 2e4;
fitting.burnin              = 1e4;               % adaptation during burn-in only, then frozen
fitting.thinning            = 10;
fitting.repetition          = 4;                 % 4 chains for R-hat
fitting.overdisp            = 0.01;              % over-dispersed chain starts
fitting.metric              = {'median','iqr'};
fitting.parameterTransform  = 'sigmoid';         % sample inside [lb,ub] without rejection
fitting.adaptStepSize       = true;
fitting.adaptCovariance     = true;              % adaptive (correlated) proposal
fitting.likelihood          = 'marginal_S0noise_flat';     % noise and M0 integrated out analytically
fitting.S0Param             = 'M0';
% hierarchical prior: R1 and R2* (M0 is integrated out)
fitting.prior               = [];
fitting.prior.hierarchical  = struct('params', {{'R1','R2star'}}, 'K', 2);
% K = 2: a two-group (mixture) population prior, more robust than K = 1 for tissue with
% atypical values (e.g. iron-rich deep grey matter)

%% Demo #1: BSP (hierarchical prior)
objGPU      = gpuJointR1R2starMapping(te,tr,fa);
% reproducibility
seed = 892396; rng(seed); gpurng(seed);
out_bsp     = objGPU.estimate(img, mask, extraData, fitting);

% population prior (in the transformed space of each parameter) and convergence
disp(out_bsp.hyper.mean.mu);
disp(out_bsp.hyper.rhat);

%% Demo #2: BSP + spatial prior (two-stage empirical Bayes)
% tau sets the strength of the spatial coupling (smaller = stronger smoothing); tau = 1
% over-smoothed in our semi-synthetic tests, tau = 3 kept the edges
fitting.prior.mrf           = struct('mode', '3d', 'tau', 3);      % 6-neighbour 3D MRF, L1 potential

objGPU      = gpuJointR1R2starMapping(te,tr,fa);
% reproducibility
seed = 892396; rng(seed); gpurng(seed);
out_bsp_mrf = objGPU.estimate(img, mask, extraData, fitting);

% stage-1 population prior (used as the fixed prior of stage 2)
disp(out_bsp_mrf.stage1.hyper.rhat);
