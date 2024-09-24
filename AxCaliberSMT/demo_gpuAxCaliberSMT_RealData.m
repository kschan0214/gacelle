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
%
%% add paths
addpath(genpath('/autofs/space/linen_001/users/kwokshing/tools/askadam')); % this path should be accessible to the group
clear;

%% I/O: Load data
data_dir = '/autofs/cluster/connectome2/Bay8_C2/bids/derivatives/processed_dwi/sub-010';

% Nb: # of unique b-value per little delta per big delta
% Nd: # of unique little delta
% ND: # of unique big delta

dwi     = niftiread(fullfile(data_dir, 'sub-010_dwi.nii.gz'));                      % full DWI data 
mask    = niftiread(fullfile(data_dir, 'sub-010_brain_mask.nii.gz'))>0;             % signal mask
bval    = readmatrix(fullfile(data_dir,'sub-010.bval'),         'FileType','text'); % 1x(Nb*Nd*ND) b-values, same length as the 4th dimension dwi
bvec    = readmatrix(fullfile(data_dir,'sub-010.bvec'),         'FileType','text'); % 3x(Nb*Nd*ND) gradient directions, 2nd dimension has the same length as the 4th dimension dwi
ldelta  = readmatrix(fullfile(data_dir,'sub-010.pulseWidth'),   'FileType','text'); % 1x(Nb*Nd*ND) little delta, same length as the 4th dimension dwi
BDELTA  = readmatrix(fullfile(data_dir,'sub-010.diffusionTime'),'FileType','text'); % 1x(Nb*Nd*ND) big delta, same length as the 4th dimension dwi

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
fitting.Nepoch              = 4000;
fitting.initialLearnRate    = 0.001;
fitting.decayRate           = 0;
fitting.convergenceValue    = 1e-8;
fitting.tol                 = 1e-3;
fitting.display             = false;
fitting.lambda              = 0.0001;
fitting.TVmode              = '3D';
fitting.voxelSize           = [2,2,2];
fitting.regmap              = 'a';
fitting.isPrior             = 1;

extractdata.bval    = bval;
extractdata.bvec    = bvec;
extractdata.ldelta  = ldelta;
extractdata.BDELTA  = BDELTA;

% intiate MCMC object
smt_gpu                     = gpuAxCaliberSMT(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);
% MCMC estimation
[out]    = smt_gpu.estimate(dwi, mask, extractdata, fitting);

%% Usage #1: Basic default setting 
fitting                     = [];
fitting.Nepoch              = 4000;
fitting.initialLearnRate    = 0.001;
fitting.decayRate           = 0;
fitting.convergenceValue    = 1e-8;
fitting.tol                 = 1e-3;
fitting.display             = false;
fitting.regmap              = {'a','f'};
fitting.lambda              = {0.001, 0.001};
fitting.TVmode              = '3D';
fitting.voxelSize           = [2,2,2];
fitting.isPrior             = 1;

extractdata = [];

% reproducibility
seed = 892396;
rng(seed);
pars0 = smt_gpu.estimate_prior(dwi_smt,mask);

% intiate MCMC object
smt_gpu                     = gpuAxCaliberSMT(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);
% MCMC estimation
[out]    = smt_gpu.estimate(dwi_smt, mask, extractdata, fitting, pars0);

