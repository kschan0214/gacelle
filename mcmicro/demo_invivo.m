%% demo_invivo.m
%
% This demo provides several examples on the ulitisation of gpumcmicro.m 
% for parameter estimation with in vivo data
% 
% Kwok-Shing Chan 
% kchan2@mgh.harvard.edu
%
% Date created: 24 June 2026 
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
BDELTA  = readmatrix(fullfile(preproc_dir,'sub-01','sub-01_preprocessed.diffusionTime'),'FileType','text'); % 1x(Nb*Nd*ND) big delta, same length as the 4th dimension dwi

%% we only work on 1 diffusion time here
BDELTA_unique   = unique(BDELTA);
idx             = BDELTA == BDELTA_unique(1); % just use the shortest diffusion time data
dwi             = dwi(:,:,:,idx);
bval            = bval(idx);
bvec            = bvec(:,idx);

bval            = bval/1e3; % convert s/mm2 to ms/um2

%% Usage #1: Basic default setting 
dwi_smt                     = gpumcmicro(bval);

fitting                     = [];
fitting.solver              = 'askadam';
fitting                     = dwi_smt.check_set_default(fitting);
fitting.start               = 'likelihood'; 

extraData                   = [];
extraData.bval              = bval;
extraData.bvec              = bvec;

out   = dwi_smt.estimate(dwi, mask, fitting,extraData);
