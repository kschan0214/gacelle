%% demo_gpuAxCaliberSMT_invivoData.m
%
% This demo provides several examples on the ulitisation of gpuAxCaliberSMT.m 
% for parameter estimation with in vivo data
% 
% Kwok-Shing Chan 
% kchan2@mgh.harvard.edu
%
% Date created: 25 March 2024 
% Date modified: 15 August 2024
% Date modified: 24 September 2024
% Date modified: 23 June 2026
%
%% add paths
addpath('../../gacelle'); addpath_gacelle; % this is the path to 'gacelle' package
clear;

%% I/O: Load data
check_dwi_invivo_demo_data; % check if the demo data exists

preproc_dir = fullfile(dwi_invivo_dir,'derivatives','preprocessed_dwi');

% Nb: # of unique b-value per little delta per big delta
% Nd: # of unique little delta
% ND: # of unique big delta
dwi     = niftiread(fullfile(preproc_dir,'sub-01','sub-01_preprocessed_dwi.nii.gz'));   % full DWI data 
mask    = dwi(:,:,:,1)>0;                                                               % signal mask
bval    = readmatrix(fullfile(preproc_dir,'sub-01','sub-01_preprocessed_dwi.bval'),'FileType','text');      % 1x(Nb*Nd*ND) b-values, same length as the 4th dimension dwi
bvec    = readmatrix(fullfile(preproc_dir,'sub-01','sub-01_preprocessed_dwi.bvec'),'FileType','text');      % 3x(Nb*Nd*ND) gradient directions, 2nd dimension has the same length as the 4th dimension dwi
ldelta  = readmatrix(fullfile(preproc_dir,'sub-01','sub-01_preprocessed.pulseWidth'),'FileType','text');    % 1x(Nb*Nd*ND) little delta, same length as the 4th dimension dwi
BDELTA  = readmatrix(fullfile(preproc_dir,'sub-01','sub-01_preprocessed.diffusionTime'),'FileType','text'); % 1x(Nb*Nd*ND) big delta, same length as the 4th dimension dwi

%% Algorithm parameters
bval        = bval/1e3; % convert s/mm2 to ms/um2
% fix tissue parameters
D0          = 1.7;
Da_fixed    = 1.7;
DeL_fixed   = 1.7;
Dcsf        = 3;

% intiate optimisation object
dwi_smt     = gpuAxCaliberSMT(bval,ldelta,BDELTA, D0, Da_fixed, DeL_fixed, Dcsf);

% set up extra data for spherical mean signal computation
extractdata.bval    = bval;
extractdata.bvec    = bvec;
extractdata.ldelta  = ldelta;
extractdata.BDELTA  = BDELTA;

%% Usage #1: Basic default setting 
fitting                     = [];
fitting.solver              = 'askadam';
fitting                     = dwi_smt.check_set_default(fitting);
fitting.start               = 'likelihood';

% reproducibility
seed = 892396; rng(seed); gpurng(seed);
% askadam estimation
out_adam = dwi_smt.estimate(dwi, mask, extractdata, fitting);

%% Usage #2: Applying spatial regularisation
fitting                     = [];
fitting.solver              = 'askadam';
fitting                     = dwi_smt.check_set_default(fitting);
fitting.start               = 'likelihood';
fitting.regmap              = {'a','f'};        % apply TV regularisation on 2 maps
fitting.lambda              = {0.0001, 0.0001};
fitting.TVmode              = '3D';
fitting.voxelSize           = [2,2,2];

% reproducibility
seed = 892396; rng(seed); gpurng(seed);
% askadam estimation
out_3DTV = dwi_smt.estimate(dwi, mask, extractdata, fitting);

%% Usage #3: MCMC Metropolis-Hasting 
fitting                     = [];
fitting.solver              = 'mcmc';
fitting                     = dwi_smt.check_set_default(fitting);
fitting.start               = 'likelihood';
fitting.iteration           = 1e6;
fitting.thinning            = 10;        % Sample every 10 iteration
fitting.metric              = {'median','iqr'};

% reproducibility
seed = 892396; rng(seed); gpurng(seed);
% askadam estimation
out_mh = dwi_smt.estimate(dwi, mask, extractdata, fitting);

%% Usage #4: MCMC Affine-invariant ensemble
fitting                     = [];
fitting.solver              = 'mcmc';
fitting                     = dwi_smt.check_set_default(fitting);
fitting.start               = 'likelihood';
fitting.algorithm           = 'ensemble';
fitting.Nwalker             = 30;
fitting.StepSize            = 2;
fitting.iteration           = 3e4;
fitting.thinning            = 10;        % Sample every 10 iteration
fitting.metric              = {'median','iqr'};

% reproducibility
seed = 892396; rng(seed); gpurng(seed);
% askadam estimation
out_ensemble = dwi_smt.estimate(dwi, mask, extractdata, fitting);