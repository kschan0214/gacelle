%% demo_gpuNEXI_invivo.m
%
% This demo provides several examples on the ulitisation of gpuNEXI.m 
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

bval = bval/1e3;

%% for NEXI we will not use b<2000 data
idx     = or(bval == 0, bval >2);
dwi     = dwi(:,:,:,idx);
bval    = bval(idx);
bvec    = bvec(:,idx);
ldelta  = ldelta(idx);
BDELTA  = BDELTA(idx);

% intiate optimisation object
dwi_smt     = gpuNEXI(bval,BDELTA);

% set up extra data for spherical mean signal computation
extraData.bval    = bval;
extraData.bvec    = bvec;
extraData.ldelta  = ldelta;
extraData.BDELTA  = BDELTA;

%% Demo#1: askadam estimation
fitting                     = [];
fitting.solver              = 'askadam';
fitting                     = dwi_smt.check_set_default(fitting);
fitting.lmax                = 2;  
fitting.start               = 'likelihood'; 

% reproducibility
seed = 892396; rng(seed); gpurng(seed);
out_adam                    = dwi_smt.estimate(dwi, mask, extraData, fitting);

%% Demo#2: mcmc estimation
% reset class object
dwi_smt     = gpuNEXI(bval,BDELTA);

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
out_ensemble = dwi_smt.estimate(dwi, mask, extraData, fitting);
