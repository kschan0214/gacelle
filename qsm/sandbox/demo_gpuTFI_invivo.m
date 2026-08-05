clear

project_dir                 = '/autofs/space/virtuoso_001/users/kwokshing/project/gacelle/';
bids_dir                    = fullfile(project_dir,'bids');
derivatives_dir             = fullfile(bids_dir,'derivatives');
sepia_dir                   = fullfile(derivatives_dir,'sepia','sourcedata-QSM_Consensus_Paper_Example_Data_v0.2.1');

sourcedata_dir              = fullfile(project_dir,'sourcedata','external','QSM_Consensus_Paper_Example_Data_v0.2.1');
sourcederivatives_dir       = fullfile(sourcedata_dir,'derivatives');
sourcesepia_dir             = fullfile(sourcederivatives_dir,'SEPIA');

work_dir        = fullfile(sourcesepia_dir,'SIEMENS','Monopolar','GRE');

subj_label = 'sub-001';
%% load data
% magnitude nifti image filename
magn_fn         = dir(fullfile(work_dir,strcat('*part-mag*')));
phas_fn         = dir(fullfile(work_dir,strcat('*part-phase*')));
sepia_header_fn = dir(fullfile(work_dir,strcat('*sepia_header.mat')));

phas = niftiread(fullfile(phas_fn.folder, phas_fn.name));
phas = (phas-min(phas(:))) ./ (max(phas(:) - min(phas(:)))) * 2*pi - pi;

img             = niftiread(fullfile(magn_fn.folder, magn_fn.name)) .*exp(1i*phas);
sepia_header    = load(fullfile(sepia_header_fn.folder, sepia_header_fn.name));

te = sepia_header.TE;

mask_lf_fn         = dir(fullfile(sepia_dir,subj_label,strcat(subj_label,'*mask_brain*.nii*')));
mask_lf_filename   = fullfile(mask_lf_fn.folder, mask_lf_fn.name);
mask_lf            = niftiread(mask_lf_filename);

mask_reliable_fn         = dir(fullfile(sepia_dir,subj_label,strcat(subj_label,'*mask_reliable*.nii*')));
mask_reliable_filename   = fullfile(mask_reliable_fn.folder, mask_reliable_fn.name);
mask_reliable            = niftiread(mask_reliable_filename);

mask = imerode(imfill( and(mask_lf,mask_reliable), 'holes'), strel('sphere',5));

% load other data
totalField_fn   = dir(fullfile(sepia_dir,subj_label,strcat('*fieldmap.nii.gz*')));
totalField      = niftiread(fullfile(totalField_fn.folder, totalField_fn.name)) ;

weights_fn   = dir(fullfile(sepia_dir,subj_label,strcat('*_weights.nii.gz*')));
weights      = niftiread(fullfile(weights_fn.folder, weights_fn.name)) .* mask;
weights      = min(weights,1);


[r2s,~,s0] = R2star_trapezoidal(abs(img),te);
img_hat     = s0 .*exp(-r2s .* permute(te(:),[2 3 4 1]));
residuals   = std(abs(img)-img_hat,[],4);
snrmap      = s0 ./ medfilt3(residuals,[5 5 5]);

weights                     = snrmap ./ mean(snrmap(mask));
weights(~isfinite(weights)) = 0;
weights                     = min(weights,2);

img_hat = s0 .*exp((-r2s + 1i*2*pi*totalField) .* permute(te(:),[2 3 4 1]));

residuals = img./img(:,:,:,1) - img_hat./img_hat(:,:,:,1);
residuals = residuals(:,:,:,2:end);
residuals = std(cat(4,real(residuals),imag(residuals)),[],4);
snrmap    = s0 ./ residuals;
snrmap(~isfinite(snrmap)) = 0;
snrmap = medfilt3(snrmap);

weights                     = snrmap ./ mean(snrmap(mask));
weights(~isfinite(weights)) = 0;
weights                     = min(weights,1);


% img_hat = abs(img) .* exp(1i*2*pi*totalField.* permute(te(:),[2 3 4 1]));
% 
% residuals = img./img(:,:,:,1) - img_hat./img_hat(:,:,:,1);
% residuals = std(residuals,[],4);
% 
% residuals_fn   = dir(fullfile(sepia_dir,subj_label,strcat('*residual.nii.gz*')));
% residuals      = niftiread(fullfile(residuals_fn.folder, residuals_fn.name));
% residuals      = min(residuals,1);

%% DEMO #1: Linear PDF

objGPU = gpuTFI(sepia_header.voxelSize, sepia_header.B0, sepia_header.B0_dir,sepia_header.TE);
% objGPU.Ps = 100;


fitting                     = [];
fitting.tol                 = 1e-6;
fitting.initialLearnRate    = 0.01;
fitting.decayRate           = 0.001;
fitting.convergenceValue    = 1e-4;

extraData                   = [];
extraData.img               = img;
extraData.weights           = weights;

fitting.lambdaTV    = 0.01;
fitting.lambdaCSF   = fitting.lambdaTV*10;
fitting.precond     = 'empirical';

fitting.iteration           = 1000;
fitting.lossFunction        = 'l2';
fitting.convergenceValue    = 1e-6;
fitting.lambdaTV            = 1e-3;
fitting.lambdaCSF           = fitting.lambdaTV*10;
fitting.initialLearnRate    = 0.001;
% fitting.decayRate           = 0.0001;
% fitting.convergenceGradTol  = 1e-8;
fitting.convergenceStepTol  = 1e-3;
% fitting.isDisplay = 1;
fitting.isnonlinear = 1;

out = objGPU.estimate(totalField,mask,extraData,fitting);

