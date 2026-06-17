addpath(genpath('../gacelle'))
addpath('/autofs/space/linen_001/users/kwokshing/tools/sepia/sepia_master');
sepia_addpath;

%% Subject info and directories
subj_label = 'sub-ms007';

bids_dir            = '/autofs/cluster/connectome2/Bay8_C2/bids/';
derivatives_dir     = fullfile(bids_dir,'derivatives/');

processed_vibe_dir      = fullfile(derivatives_dir,'preprocessed',subj_label,'anat');

file_list = dir(fullfile(processed_vibe_dir,'*rec-RR*space-MEGRE_sepia_header.mat'));
flip_angle = zeros(1,numel(file_list));
for kfile = 1:numel(file_list)
    load(fullfile(file_list(kfile).folder,file_list(kfile).name),'FA');
    flip_angle(kfile) = FA;
end
flip_angle = sort(flip_angle,'ascend');

%% load data
counter         = 0;
img             = [];
sepia_header    = [];
unwrappedPhase  = [];
totalField      = [];
fa              = zeros(1,length(flip_angle));
for kfa = 1:length(flip_angle)
    counter = counter + 1;
    
    FAcurr          = sprintf('%02d', flip_angle(kfa));

     % general GRE basename
    acq_label   = strcat('acq-',['FA' FAcurr]);
    prefix      = strcat(subj_label,'_',acq_label,'_rec-RR');

    % magnitude nifti image filename
    magn_fn         = dir(fullfile(processed_vibe_dir,strcat(prefix,'*part-mag*space-MEGRE_MEGRE_preprocessed.nii*')));
    sepia_header_fn = dir(fullfile(processed_vibe_dir,strcat(prefix,'*space-MEGRE_sepia_header.mat')));

    nii                 = load_untouch_nii(fullfile(magn_fn.folder, magn_fn.name));
    img                 = cat(5,img,nii.img);
    sepia_header{kfa}   = load(fullfile(sepia_header_fn.folder, sepia_header_fn.name));

    fa(kfa)  = sepia_header{kfa}.FA;
    tr      = sepia_header{kfa}.TR;

end
te = sepia_header{end}.TE;

% B1 info
true_flip_angle_fn      = dir(fullfile(processed_vibe_dir,'*acq-famp*TB1TFL*space-MEGRE_RR.nii*'));
true_flip_angle_json    = dir(fullfile(processed_vibe_dir,'*acq-famp*TB1TFL*.json'));

true_flip_angle         = load_nii_img_only( fullfile( true_flip_angle_fn.folder, true_flip_angle_fn.name));
b1_header               = jsondecode( fileread( fullfile( true_flip_angle_json.folder,true_flip_angle_json.name)));

b1                      = true_flip_angle / 10 / b1_header.FlipAngle;

mask_fn         = dir(fullfile(processed_vibe_dir,strcat(subj_label,'*acq-FA10*rec-RR*mask_brain*.nii*')));
mask_filename   = fullfile(mask_fn.folder, mask_fn.name);
mask            = load_nii_img_only(mask_filename);

clear true_flip_angle

extradata       = [];
extradata.b1    = b1;
%% Prepare data fot batch processing
% Magnitude fitting
% setup algorithm parameters
fitting                     = [];
fitting.solver              = 'askadam';
fitting                     = objGPU.check_set_default(fitting);

obj     = gpuJointR1R2starMapping(te,tr,fa);
[out]   = obj.estimate(img, mask, extradata, fitting);

% %% MCMC
% % Magnitude fitting
% % setup algorithm parameters
% fitting                     = [];
% fitting.solver              = 'mcmc';
% fitting                     = objGPU.check_set_default(fitting);
% fitting.algorithm           = 'ensemble';
% fitting.iteration           = 1e4;
% fitting.thinning            = 10;        % Sample every 20 iteration
% fitting.metric              = {'median','iqr'};
% fitting.burnin              = 0.1;       % 10% burn-in
% fitting.Nwalker             = 30;
% 
% obj         = gpuJointR1R2starMapping(te,tr,fa);
% [out_mcmc]  = obj.estimate(img, mask, extradata, fitting);
