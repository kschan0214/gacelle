clear

project_dir                 = '/autofs/space/virtuoso_001/users/kwokshing/project/gacelle/';
bids_dir                    = fullfile(project_dir,'bids');
derivatives_dir             = fullfile(bids_dir,'derivatives');
sepia_dir                   = fullfile(derivatives_dir,'sepia','sourcedata-dsc-3015069.02/');

sourcedata_dir              = fullfile(project_dir,'sourcedata','external','dsc-3015069.02/');

pipeline = 'ROMEOPDFFANSI';

% Optional user input 
% Input/Output filenames
% You can also specify the directory containing the SEPIA-ready data here

input           = [];
% mask_filename   = fullfile(sourcedata_dir,'Simdata','ChallengeProtocol','Simulated_1p0mm','Brain.nii.gz');
% input(1).name   = fullfile(sourcedata_dir,'Simdata','ChallengeProtocol','Simulated_1p0mm','Data_phase.nii.gz');
% input(2).name   = fullfile(sourcedata_dir,'Simdata','ChallengeProtocol','Simulated_1p0mm','Data_magn.nii.gz');
input(1).name   = fullfile(sourcedata_dir,'ReleaseDraftChallenge','Sim2','Phase.nii.gz');
input(2).name   = fullfile(sourcedata_dir,'ReleaseDraftChallenge','Sim2','Magnitude.nii.gz');
input(3).name   = [];
input(4).name   = fullfile(sepia_dir,'sepia_header.mat');
mask_filename   = fullfile(sourcedata_dir,'ReleaseDraftChallenge','Sim2','MaskBrainExtracted.nii.gz');
output_basename = fullfile(sepia_dir,pipeline,'sub-001','sub-001');

data_mag    = niftiread(input(2).name);
data_phase  = niftiread(input(1).name);
mask        = niftiread(mask_filename);
% totalField  = niftiread(fullfile(sepia_dir,pipeline,'sub-001','sub-001_fieldmap.nii.gz'));
load(input(4).name);
mask_romeo  = niftiread(fullfile(sepia_dir,pipeline,'sub-001','sub-001_mask_localfield.nii.gz'));
totalField = zeros(size(mask));

% chi_GT      = niftiread(fullfile(sourcedata_dir,'Simdata','ChallengeProtocol','Simulated_1p0mm','Chi_cropBrainExtracted.nii.gz'));
chi_GT      = niftiread(fullfile(sourcedata_dir,'ReleaseDraftChallenge','Sim2GT','Chi.nii.gz'));

data = data_mag .* exp(1i*data_phase);

%%
objGPU = gpumcTFI(voxelSize, B0, B0_dir,TE);
% objGPU.Ps = 100;

fitting                     = [];
fitting.tol                 = 1e-6;
fitting.initialLearnRate    = 0.01;
fitting.decayRate           = 0.001;
fitting.convergenceValue    = 1e-4;

extraData                   = [];
extraData.weights           = mask;
extraData.fint              = totalField;

fitting.precond     = 'none';

fitting.iteration           = 10000;
fitting.lossFunction        = 'l2';
fitting.convergenceValue    = 1e-5;
fitting.convergenceGradTol  = 1e-6;
fitting.convergenceStepTol  = 1e-6;
fitting.lambdaTV            = 1e-2;
fitting.lambdaCSF           = fitting.lambdaTV*10;%fitting.lambdaTV*10;
% fitting.lambdaTNV           = 0;
fitting.initialLearnRate    = 0.001;
fitting.decayRate           = 0.001;
fitting.isDisplay           = 0;
fitting.start               = 'prior';
fitting.patience            = 1;
out = objGPU.estimate(data,mask,extraData,fitting);
% out = objGPU.estimate(data,mask_romeo,extraData,fitting);

%%
% mask_metric = and(mask_qsm,mask);
mask_metric = mask;
% 
% segmentation = niftiread(fullfile(sourcedata_dir,'Simdata','ChallengeProtocol','Simulated_1p0mm','FinalSegment.nii.gz'));
% load(fullfile(sourcedata_dir,'Simdata','ChallengeProtocol','Simulated_1p0mm','FinalSegment.nii.gz'),'label');

segmentation = niftiread(fullfile(sourcedata_dir,'ReleaseDraftChallenge','Sim2GT','Segmentation.nii.gz'));
load(fullfile(sourcedata_dir,'ReleaseDraftChallenge','Sim2GT','label.mat'),'label');
load(fullfile(sourcedata_dir,'ReleaseDraftChallenge','Sim2GT','FilestructureForEval.mat'));
% Tissue
msk2_tissue = mask_metric;
msk2_tissue( or(segmentation < 7 , segmentation > 9 ) ) = 0 ;
msk2_tissue = 1-dilatemask(1-msk2_tissue,1) ; % erodes tissue mask
% Tissue
msk2_vessel = mask_metric;
msk2_vessel( (segmentation ~= 11) ) = 0;
msk2_vessel=dilatemask(msk2_vessel,1); % dilates vein mask
% DGM
msk2_dgm = mask_metric;
msk2_dgm( (segmentation >= 7) ) = 0;
msk2_dgm = 1-dilatemask( 1 - msk2_dgm , 1) ; % erodes deep grey matter 

%% run metrics
addpath('/autofs/space/symphony_002/users/kwokshing/external_data/dsc3015069.02/ReleaseDraftChallenge/functions4challenge/evaluation');
chi_input = out.final.chi;
metrics = [];
[metrics.rmse, metrics.rmse_detrend] = compute_rmse_detrend_v1(chi_input,chi_GT, mask_metric);

% Tissue
[~, metrics.rmse_detrend_Tissue]    = compute_rmse_detrend_v1( chi_input, chi_GT, msk2_tissue );

% Vessels Only
[~, metrics.rmse_detrend_Blood]     = compute_rmse_detrend_v1( chi_input, chi_GT, msk2_vessel );

 %  Deep Gray Matter
[~, metrics.rmse_detrend_DGM]       = compute_rmse_detrend_v1( chi_input, chi_GT, msk2_dgm );
[DGM_slope_ds]                      = compute_linearityDeepGM( chi_input , chi_GT , segmentation , label);
metrics.DeviationFromLinearSlope    = abs(1-DGM_slope_ds) ;

% Calcification metrics
[ CalcMoment , metrics.CalcStreak, metrics.CalcStreakNotNorm, metrics.CalcStreakInOut] = compute_calcification_metrics_v2 ( chi_input , chi_GT , segmentation , label );

%  to ensure all the metrics have to be minimized, we calculate the
%  difference between calcification obtained from image and ground truth
metrics.DeviationFromCalcMoment =  abs(filesstructure.CalcMoment - CalcMoment) 

%%
chi_standard = niftiread(fullfile(sepia_dir,'ROMEOPDFFANSI','Sim2','Sim2_Chimap.nii.gz'));
chi_input = chi_standard;
metrics = [];
[metrics.rmse, metrics.rmse_detrend] = compute_rmse_detrend_v1(chi_input,chi_GT, mask);

% Tissue
[~, metrics.rmse_detrend_Tissue]    = compute_rmse_detrend_v1( chi_input, chi_GT, msk2_tissue );

% Vessels Only
[~, metrics.rmse_detrend_Blood]     = compute_rmse_detrend_v1( chi_input, chi_GT, msk2_vessel );

 %  Deep Gray Matter
[~, metrics.rmse_detrend_DGM]       = compute_rmse_detrend_v1( chi_input, chi_GT, msk2_dgm );
[DGM_slope_ds]                      = compute_linearityDeepGM( chi_input , chi_GT , segmentation , label);
metrics.DeviationFromLinearSlope    = abs(1-DGM_slope_ds) ;

% Calcification metrics
[ CalcMoment , metrics.CalcStreak, metrics.CalcStreakNotNorm, metrics.CalcStreakInOut] = compute_calcification_metrics_v2 ( chi_input , chi_GT , segmentation , label );

%  to ensure all the metrics have to be minimized, we calculate the
%  difference between calcification obtained from image and ground truth
metrics.DeviationFromCalcMoment =  abs(filesstructure.CalcMoment - CalcMoment) 
