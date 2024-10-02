addpath(genpath('/autofs/space/linen_001/users/kwokshing/tools/askAdam'))
addpath('/autofs/space/linen_001/users/kwokshing/tools/sepia/sepia_master');
sepia_addpath;

%% Subject info and directories
subj_label = 'sub-022';

bids_dir            = '/autofs/cluster/connectome2/Bay8_C2/bids/';
derivatives_dir     = fullfile(bids_dir,'derivatives/');

jointR1R2star_dir       = fullfile(derivatives_dir,'jointR1R2star',subj_label); 
SEPIA_dir               = fullfile(derivatives_dir,'sepia',subj_label);
processed_vibe_dir      = fullfile(derivatives_dir,'processed_vibe',subj_label);

file_list = dir(fullfile(processed_vibe_dir,'*rec-offline*space-MEGRE.mat'));
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
    
     % general GRE basename
    acq_label   = strcat('acq-',['FA' num2str(flip_angle(kfa))]);
    prefix      = strcat(subj_label,'_',acq_label,'_rec-offline');

    % magnitude nifti image filename
    magn_fn         = dir(fullfile(processed_vibe_dir,strcat(prefix,'*part-mag*MEGRE_space-MEGRE.nii*')));
    sepia_header_fn = dir(fullfile(processed_vibe_dir,strcat(prefix,'*space-MEGRE.mat')));

    nii                 = load_untouch_nii(fullfile(magn_fn.folder, magn_fn.name));
    img                 = cat(5,img,nii.img);
    sepia_header{kfa}   = load(fullfile(sepia_header_fn.folder, sepia_header_fn.name));

    fa(kfa)  = sepia_header{kfa}.FA;
    tr      = sepia_header{kfa}.TR;

end
te = sepia_header{end}.TE;

% B1 info
true_flip_angle_fn      = dir(fullfile(processed_vibe_dir,'*acq-famp*TB1TFL*space-MEGRE.nii*'));
true_flip_angle_json    = dir(fullfile(processed_vibe_dir,'*acq-famp*TB1TFL*.json'));

true_flip_angle         = load_nii_img_only( fullfile( true_flip_angle_fn.folder, true_flip_angle_fn.name));
b1_header               = jsondecode( fileread( fullfile( true_flip_angle_json.folder,true_flip_angle_json.name)));

b1                      = true_flip_angle / 10 / b1_header.FlipAngle;

mask_fn         = dir(fullfile(processed_vibe_dir,strcat(subj_label,'*acq-FA10*rec-offline*mask_brain*space-MEGRE.nii*')));
mask_filename   = fullfile(mask_fn.folder, mask_fn.name);
mask            = load_nii_img_only(mask_filename);

clear true_flip_angle

%% Prepare data fot batch processing
% Magnitude fitting
% setup algorithm parameters
fitting                 = [];
fitting.algorithm       = 'GW';
fitting.Nwalker         = 50;
fitting.StepSize        = 2;
fitting.iteration       = 1e4;
fitting.thinning        = 10;        % Sample every 20 iteration
fitting.metric          = {'median','iqr'};
fitting.burnin          = 0.1;       % 10% burn-in
% residual option
fitting.isWeighted      = true;
fitting.weightPower     = 2;
fitting.weightMethod    = '1stecho';
extradata = [];
extradata.b1 = b1;

obj     = gpuJointR1R2starMappingmcmc(te,tr,fa);
[out]   = obj.estimate(img, mask, extradata, fitting);
