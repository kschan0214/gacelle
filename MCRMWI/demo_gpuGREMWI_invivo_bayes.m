%% demo_gpuGREMWI_invivo_bayes.m
%
% This demo shows the Bayesian priors of the EXPERIMENTAL mcmc_bayes sampler with gpuGREMWI.m
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
%   estimate() stops with the error gpuGREMWI:singleSegment).
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
kfa             = 3;
FAcurr          = sprintf('%d', flip_angle(kfa));
acq_label       = strcat('acq-',['TR50NTE15FA' FAcurr]);
prefix          = strcat(subj_label,'_',sess_label,'_',acq_label);

magn_fn         = dir(fullfile(preproc_dir,subj_label,sess_label,'anat',strcat(prefix,'_*part-mag*_MEGRE_space-withinGRE.nii.gz*')));
phas_fn         = dir(fullfile(preproc_dir,subj_label,sess_label,'anat',strcat(prefix,'_*part-phase*_MEGRE_space-withinGRE.nii.gz*')));
sepia_header_fn = dir(fullfile(preproc_dir,subj_label,sess_label,'anat',strcat(prefix,'_*MEGRE_sepia_header.mat')));
img             = niftiread(fullfile(magn_fn.folder, magn_fn.name)) .*exp(1i*niftiread(fullfile(phas_fn.folder, phas_fn.name)));
sepia_header    = load(fullfile(sepia_header_fn.folder, sepia_header_fn.name));
te              = sepia_header.TE;

mask_fn         = dir(fullfile(preproc_dir,subj_label,sess_label,'anat',strcat(subj_label,'*brain_mask*.nii*')));
mask            = niftiread(fullfile(mask_fn.folder, mask_fn.name)) > 0;

totalField_fn   = dir(fullfile(sepia_dir,subj_label,sess_label,'anat',strcat('FA',FAcurr),strcat(prefix,'_*fieldmap.nii.gz*')));
totalField      = niftiread(fullfile(totalField_fn.folder, totalField_fn.name)) ;

% estimate initial phase
pini            = angle(img(:,:,:,1) ./ exp(1i* 2*pi*totalField .* permute(te(1),[2 3 4 1])));

%% fixed parameters
kappa_mw                = 0.36; % Jung, NI., myelin water density
kappa_iew               = 0.86; % Jung, NI., intra-/extra-axonal water density
fixed_params.B0     	= 3;    % field strength, in tesla
fixed_params.rho_mw    	= kappa_mw/kappa_iew; % relative myelin water density
fixed_params.E      	= 0.02; % exchange effect in signal phase, in ppm
fixed_params.x_i      	= -0.1; % myelin isotropic susceptibility, in ppm
fixed_params.x_a      	= -0.1; % myelin anisotropic susceptibility, in ppm
fixed_params.B0dir      = sepia_header.B0_dir;

%% slab of slices (see Notes)
slab                = round(size(mask,3)/2) + (-4:3);
img                 = img(:,:,slab,:);
mask                = mask(:,:,slab);
extraData           = [];
extraData.freqBKG   = totalField(:,:,slab) / (gpuGREMWI.gyro*fixed_params.B0); % in ppm
extraData.pini      = pini(:,:,slab);

%% Bayesian MCMC settings shared by both demos
fitting                     = [];
fitting.solver              = 'mcmc';
objGPU                      = gpuGREMWI(te,fixed_params);
fitting                     = objGPU.check_set_default(fitting,img);
fitting.start               = 'prior';
fitting.weightPower         = 0.5;
% conventional GRE-MWI (all compartmental parameters free)
fitting.DIMWI.isFitIWF      = 1;
fitting.DIMWI.isFitFreqMW   = 1;
fitting.DIMWI.isFitFreqIW   = 1;
fitting.DIMWI.isFitR2sEW    = 1;
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
fitting.likelihood          = 'marginal_noise';            % noise integrated out analytically
% hierarchical prior: the tissue parameters; the background frequency and
% initial phase offsets (dfreqBKG, dpini) vary smoothly in space and keep a flat prior
fitting.prior               = [];
fitting.prior.hierarchical  = struct('params', {{'MWF','IWF','R2sMW','R2sIW','R2sEW','freqMW','freqIW'}}, 'K', 1);
% K = 2 gives a two-group (mixture) population prior if the tissue is heterogeneous

%% Demo #1: BSP (hierarchical prior)
objGPU      = gpuGREMWI(te,fixed_params);
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

objGPU      = gpuGREMWI(te,fixed_params);
% reproducibility
seed = 892396; rng(seed); gpurng(seed);
out_bsp_mrf = objGPU.estimate(img, mask, extraData, fitting);

% stage-1 population prior (used as the fixed prior of stage 2)
disp(out_bsp_mrf.stage1.hyper.rhat);
