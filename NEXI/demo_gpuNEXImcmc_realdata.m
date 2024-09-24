addpath(genpath('/autofs/space/linen_001/users/kwokshing/tools/askadam'))
clear

%% basic I/O
subj_label  ='sub-NEXIC2HC006';
reco_label  = 'recon-ORIG';
acq         = {'D13ms', 'D21ms', 'D30ms'};

model_label = 'model-nexi';
solver_label= 'solver-AskAdam';
lmax_label  = 'lmax-2';

project_dir     = '/autofs/space/symphony_002/users/kwokshing/project/nexi/';
bids_dir        = fullfile(project_dir,'bids');
derivatives_dir = fullfile(bids_dir,'derivatives');
preprocessed_dir= fullfile(derivatives_dir,'preprocessed');
nexi_dir        = fullfile(derivatives_dir,'NEXI');


%% load data
data_dir = fullfile(preprocessed_dir,subj_label,'dwi');

dwi_nii     = [];
dwi_bval    = [];
for k = 1:numel(acq)
dwi_nii{k}  = strcat(subj_label,'_',['acq-' acq{k}],'_',reco_label,"_dwi_denoise_degibbs_eddy_gncorr.nii.gz");
dwi_bval{k} = strcat(subj_label,'_',['acq-' acq{k}],'_',reco_label,"_dwi_corr_bvals.txt");
dwi_bvec{k} = strcat(subj_label,'_',['acq-' acq{k}],'_',reco_label,"_dwi.bvec");
end
% DWI data
dwi_D13 = niftiread(fullfile(data_dir,dwi_nii{1}));
dwi_D21 = niftiread(fullfile(data_dir,dwi_nii{2}));
dwi_D30 = niftiread(fullfile(data_dir,dwi_nii{3}));
% b-values
bvals_D13 = readmatrix(fullfile(data_dir,dwi_bval{1}));
bvals_D21 = readmatrix(fullfile(data_dir,dwi_bval{2}));
bvals_D30 = readmatrix(fullfile(data_dir,dwi_bval{3}));
% diffusion gradient directions
bvecs_D13 = readmatrix(fullfile(data_dir,dwi_bvec{1}),"FileType","text");
bvecs_D21 = readmatrix(fullfile(data_dir,dwi_bvec{2}),"FileType","text");
bvecs_D30 = readmatrix(fullfile(data_dir,dwi_bvec{3}),"FileType","text");
% mask
nii_info    = niftiinfo(fullfile(data_dir ,strcat(subj_label,'_',reco_label,'_dwi_brain_mask_gncorr.nii.gz')));
mask        = niftiread(nii_info)>0;

%% prepare data for data fitting
% concatenate all data into a single variable
dwi     = cat(4,dwi_D13, dwi_D21, dwi_D30);
bvals   = cat(2,bvals_D13, bvals_D21, bvals_D30) / 1e3;
bvecs   = cat(2,bvecs_D13, bvecs_D21, bvecs_D30);
ldelta  = cat(2,ones(size(bvals_D13))*6, ones(size(bvals_D21))*6, ones(size(bvals_D30))*6);     % diffusion gradient pulse width, in ms
BDELTA  = cat(2,ones(size(bvals_D13))*13, ones(size(bvals_D21))*21, ones(size(bvals_D30))*30);  % diffusion time, in ms
% exclude b=1000 data as descibed in the manuscript
idx         = or(bvals > 1, bvals == 0);
dwi_bgt1    = dwi(:,:,:,idx);
bvals_bgt1  = bvals(:,idx);
bvecs_bgt1  = bvecs(:,idx);
ldelta_bgt1 = ldelta(:,idx);
BDELTA_bgt1 = BDELTA(:,idx);

clear dwi_D13 dwi_D21 dwi_D30 dwi 

obj = DWIutility;
% get unique bvals
[bval_fit,ldelta_fit,BDELTA_fit] = obj.unique_shell(bvals_bgt1,ldelta_bgt1,BDELTA_bgt1);
% compute lmax = 4
dwi_bgt1 = obj.get_Sl_all(dwi_bgt1,bvals_bgt1,bvecs_bgt1,ldelta_bgt1,BDELTA_bgt1,4);
% only get lmax = 2
dwi_bgt1 = dwi_bgt1(:,:,:,1:numel(bval_fit)*2);

% create fit object
nexi_obj = gpuNEXImcmc(bval_fit, BDELTA_fit);

%% Model fitting
fitting             = [];
fitting.algorithm   = 'MH';     % Metropolis-Hastings
fitting.iteration   = 2e4;      % 2e4 for demo purpose. Original implementation used 2e5.
fitting.thinning    = 100;
fitting.burnin      = 0.1;
fitting.method      = 'median';
fitting.start       = 'default';
fitting.lmax        = 2;

% Optional, in case the input dwi is full dataset
extradata.bval      = bvals_bgt1;
extradata.bvec      = bvecs_bgt1;
extradata.ldelta    = ldelta_bgt1;
extradata.BDELTA    = BDELTA_bgt1;

disp(subj_label);

% askadam optimisation
[out] = nexi_obj.estimate(dwi_bgt1, mask, extradata, fitting);

%% Model fitting
fitting             = [];
fitting.algorithm   = 'GW';     % Metropolis-Hastings
fitting.Nwalker     = 40;
fitting.iteration   = 2e3;      % 2e4 for demo purpose. Original implementation used 2e5.
fitting.thinning    = 10;
fitting.burnin      = 0.2;
fitting.method      = 'median';
fitting.start       = 'likelihood';
fitting.lmax        = 2;

% Optional, in case the input dwi is full dataset
extradata.bval      = bvals_bgt1;
extradata.bvec      = bvecs_bgt1;
extradata.ldelta    = ldelta_bgt1;
extradata.BDELTA    = BDELTA_bgt1;

disp(subj_label);

% askadam optimisation
nexi_obj = gpuNEXImcmc(bval_fit, BDELTA_fit);
[out] = nexi_obj.estimate(dwi_bgt1, mask, extradata, fitting);

