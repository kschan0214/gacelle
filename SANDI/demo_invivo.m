%% demo_invivo.m
%
% This demo provides several examples on the ulitisation of gpuSANDI.m 
% for parameter estimation with in vivo data
% 
% Kwok-Shing Chan 
% kchan2@mgh.harvard.edu
%
% Date created: 24 June 2026 
%
%% add paths
addpath(genpath('../../gacelle')); % this is the path to 'gacelle' package
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

%% we only work on 1 diffusion time here
BDELTA_unique   = unique(BDELTA);
idx             = BDELTA == BDELTA_unique(1); % just use the shortest diffusion time data
dwi             = dwi(:,:,:,idx);
bval            = bval(idx);
bvec            = bvec(:,idx);
ldelta          = ldelta(idx);
BDELTA          = BDELTA(idx);

bval            = bval/1e3; % convert s/mm2 to ms/um2

% setting up extra data for rotationally invariant signal computation
extraData                   = [];
extraData.bval              = bval;
extraData.bvec              = bvec;
extraData.BDELTA            = BDELTA;
extraData.ldelta            = ldelta;

Ds = 3; % intrinsic diffusivity of soma 

%% Demo #1: askadam.m estimation
dwi_smt                     = gpuSANDI(bval,ldelta,BDELTA,Ds);

fitting                     = [];
fitting.solver              = 'askadam';
fitting                     = dwi_smt.check_set_default(fitting);
fitting.start               = 'likelihood'; 

out_askadam   = dwi_smt.estimate(dwi, mask, fitting,extraData);

%% Demo #2: mcmc.m estimation
% reset class object
dwi_smt                     = gpuSANDI(bval,ldelta,BDELTA,Ds);

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

% askadam estimation
out_ensemble = dwi_smt.estimate(dwi, mask, fitting,extraData);