%% demo_gpuAxCaliberSMTmcmc_RealData.m
%
% This demo provides several examples on the ulitisation of gpuAxCaliberSMTmcmc.m 
% for parameter estimation with in vivo data
% 
% Kwok-Shing Chan 
% kchan2@mgh.harvard.edu
%
% Date created: 25 March 2024 
% Date modified: 15 August 2024
% Date modified: 24 September 2024
%
%% add paths
addpath(genpath('../../gacelle')); % this is the path to 'gacelle' package
clear;

%% I/O: Load data
data_dir = '/path/to/your/bids/derivatives/processed_dwi/sub-<label>';

% Nb: # of unique b-value per little delta per big delta
% Nd: # of unique little delta
% ND: # of unique big delta

dwi     = niftiread(fullfile(data_dir, 'sub-<label>_dwi.nii.gz'));                      % full DWI data 
mask    = niftiread(fullfile(data_dir, 'sub-<label>_brain_mask.nii.gz'))>0;             % signal mask
bval    = readmatrix(fullfile(data_dir,'sub-<label>.bval'),         'FileType','text'); % 1x(Nb*Nd*ND) b-values, same length as the 4th dimension dwi
bvec    = readmatrix(fullfile(data_dir,'sub-<label>.bvec'),         'FileType','text'); % 3x(Nb*Nd*ND) gradient directions, 2nd dimension has the same length as the 4th dimension dwi
ldelta  = readmatrix(fullfile(data_dir,'sub-<label>.pulseWidth'),   'FileType','text'); % 1x(Nb*Nd*ND) little delta, same length as the 4th dimension dwi
BDELTA  = readmatrix(fullfile(data_dir,'sub-<label>.diffusionTime'),'FileType','text'); % 1x(Nb*Nd*ND) big delta, same length as the 4th dimension dwi

%% Algorithm parameters
bval        = bval/1e3; % s/mm2 to ms/um2
% fix parameters
D0          = 1.7;
Da_fixed    = 1.7;
DeL_fixed   = 1.7;
Dcsf        = 3;

% get unique b-values for each little delta and big delta
[bval_sorted,ldelta_sorted,BDELTA_sorted] = DWIutility.unique_shell(bval,ldelta,BDELTA);

obj     = DWIutility;
lmax    = 0;
dwi_smt = obj.get_Sl_all(dwi,bval,bvec,ldelta,BDELTA,lmax);

%% Usage #1: Basic default setting 
fitting                     = [];
fitting.itearation          = 4000;
fitting.initialLearnRate    = 0.001;
fitting.convergenceValue    = 1e-8;
fitting.tol                 = 1e-3;
fitting.isdisplay           = false;
fitting.lambda              = 0;
fitting.isPrior             = 1;

extractdata.bval    = bval;
extractdata.bvec    = bvec;
extractdata.ldelta  = ldelta;
extractdata.BDELTA  = BDELTA;

% reproducibility
seed = 892396; rng(seed); gpurng(seed);
% intiate askadam object
smt_gpu     = gpuAxCaliberSMT(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);
% askadam estimation
[out]       = smt_gpu.estimate(dwi, mask, extractdata, fitting);

%% Usage #2: Applying spatial regularisation
fitting                     = [];
fitting.itearation          = 4000;
fitting.initialLearnRate    = 0.001;
fitting.convergenceValue    = 1e-8;
fitting.tol                 = 1e-3;
fitting.isdisplay           = false;
fitting.regmap              = {'a','f'};        % apply TV regularisation on 2 maps
fitting.lambda              = {0.0001, 0.0001};
fitting.TVmode              = '3D';
fitting.voxelSize           = [2,2,2];
fitting.isPrior             = 1;

extractdata.bval    = bval;
extractdata.bvec    = bvec;
extractdata.ldelta  = ldelta;
extractdata.BDELTA  = BDELTA;

% reproducibility
seed = 892396; rng(seed); gpurng(seed);

% intiate askadam object
smt_gpu     = gpuAxCaliberSMT(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);
% askadam estimation
[out]       = smt_gpu.estimate(dwi, mask, extractdata, fitting);
