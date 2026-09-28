%% demo_gpumcmicro_invivo_bayes.m
%
% This demo shows the Bayesian priors of the EXPERIMENTAL mcmc_bayes sampler with gpumcmicro.m
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
%   estimate() stops with the error gpumcmicro:singleSegment).
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
dwi_invivo_dir = fullfile('~/Downloads','ds006181'); % this is where the data locates, feel free to update this path
check_dwi_invivo_demo_data; % check if the demo data exists, if not then download it to dwi_invivo_dir

preproc_dir = fullfile(dwi_invivo_dir,'derivatives','preprocessed_dwi');

dwi     = niftiread(fullfile(preproc_dir,'sub-01','sub-01_preprocessed_dwi.nii.gz'));   % full DWI data
mask    = dwi(:,:,:,1)>0;                                                               % signal mask
bval    = readmatrix(fullfile(preproc_dir,'sub-01','sub-01_preprocessed_dwi.bval'),'FileType','text');      % 1xNdwi b-values
bvec    = readmatrix(fullfile(preproc_dir,'sub-01','sub-01_preprocessed_dwi.bvec'),'FileType','text');      % 3xNdwi gradient directions
ldelta  = readmatrix(fullfile(preproc_dir,'sub-01','sub-01_preprocessed.pulseWidth'),'FileType','text');    % 1xNdwi little delta
BDELTA  = readmatrix(fullfile(preproc_dir,'sub-01','sub-01_preprocessed.diffusionTime'),'FileType','text'); % 1xNdwi big delta
bval    = bval/1e3; % convert s/mm2 to ms/um2

%% we only work on 1 diffusion time here
BDELTA_unique   = unique(BDELTA);
idx             = BDELTA == BDELTA_unique(1); % just use the shortest diffusion time data
dwi             = dwi(:,:,:,idx);
bval            = bval(idx);
bvec            = bvec(:,idx);

%% slab of slices (see Notes)
slab    = round(size(mask,3)/2) + (-4:3);
dwi     = dwi(:,:,slab,:);
mask    = mask(:,:,slab);

extraData                   = [];
extraData.bval              = bval;
extraData.bvec              = bvec;

%% Bayesian MCMC settings shared by both demos
fitting                     = [];
fitting.solver              = 'mcmc';
fitting.start               = 'likelihood';
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
% hierarchical prior: neurite fraction and intrinsic diffusivity
fitting.prior               = [];
fitting.prior.hierarchical  = struct('params', {{'f','D'}}, 'K', 1);
% K = 2 gives a two-group (mixture) population prior if the tissue is heterogeneous

%% Demo #1: BSP (hierarchical prior)
dwi_smt     = gpumcmicro(bval);
% reproducibility
seed = 892396; rng(seed); gpurng(seed);
out_bsp     = dwi_smt.estimate(dwi, mask, extraData, fitting);

% population prior (in the transformed space of each parameter) and convergence
disp(out_bsp.hyper.mean.mu);
disp(out_bsp.hyper.rhat);

%% Demo #2: BSP + spatial prior (two-stage empirical Bayes)
% tau sets the strength of the spatial coupling (smaller = stronger smoothing); tau = 1
% over-smoothed in our semi-synthetic tests, tau = 3 kept the edges
fitting.prior.mrf           = struct('mode', '3d', 'tau', 3);      % 6-neighbour 3D MRF, L1 potential

dwi_smt     = gpumcmicro(bval);
% reproducibility
seed = 892396; rng(seed); gpurng(seed);
out_bsp_mrf = dwi_smt.estimate(dwi, mask, extraData, fitting);

% stage-1 population prior (used as the fixed prior of stage 2)
disp(out_bsp_mrf.stage1.hyper.rhat);
