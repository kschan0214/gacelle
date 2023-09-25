
clear; close all

%% Subject info and directories
subj='002' ;
% subj	= '002';
ses='mri01' ;
run     = '1';
% ses     = 'mri01';

subj_label  = ['sub-' subj];
sess_label  = ['ses-' ses];
run_label   = ['run-' run];

% load path
addpath('/project/3015069.05/bids/code/');
load_module_despot1;
load_module_epg_epgx;
load_module_mwi;
load_module_sepia;

% call diretcories
subject_directory_master

subj_script_dir = fullfile(code_dir, subj_label, sess_label);

protocol= { 'TR38NTE12' } ;
% GRE Protocols and flip angles
% slice=1;
% protocol    = { 'TR38NTE12', 'TR50NTE15', 'TR55NTE13' };
% protocol    = { 'TR38NTE12'};
flipAngle   = { 'FA5', 'FA10', 'FA20', 'FA50', 'FA70' };

% load GRE data
for kp = protocol
    
    prot_ANTs_dir   = fullfile(ANTs_gre_within_dir, kp{1});
    prot_SEPIA_dir  = fullfile(derivative_SEPIA_dir, kp{1});

    % load GRE data
    counter         = 0;
    img             = [];
    unwrappedPhase  = [];
    totalField      = [];
    sepia_header    = [];
    fa              = zeros(1,length(flipAngle));
    for kf = flipAngle
        counter = counter + 1;
        seq_SEPIA_dir   = fullfile(prot_SEPIA_dir, kf{1});
        
        % general GRE basename
        seq             = [kp{1} kf{1}];
        acq_label       = ['acq-' seq];
        gre_basename    = [subj_label '_' sess_label '_' acq_label '_' run_label];
        
        % magnitude nifti image filename
        magn_fn         = [gre_basename '_part-mag_MEGRE_space-withinGRE.nii.gz'];
        phase_fn        = [gre_basename '_MEGRE_space-withinGRE_EchoCombine-OW_unwrapped-phase.nii.gz'];
        totalField_fn   = [gre_basename '_MEGRE_space-withinGRE_EchoCombine-OW_total-field.nii.gz'];
        
        sepia_header_fn = [gre_basename '_MEGRE_sepia_header.mat'];
        
        nii                     = load_untouch_nii(fullfile(ANTs_gre_apply_within_dir, magn_fn));
        img                     = cat(5,img,nii.img);
        nii = load_untouch_nii(fullfile(seq_SEPIA_dir, phase_fn));
        unwrappedPhase = cat(5,unwrappedPhase,nii.img);
        sepia_header{counter}	= load(fullfile(prot_SEPIA_dir, sepia_header_fn));
        
        totalField = cat(4,totalField, load_nii_img_only(fullfile(seq_SEPIA_dir, totalField_fn)));
        
        fa(counter)   	= sepia_header{counter}.FA;
        tr              = sepia_header{counter}.TR;

    end
    
    % B1 info
    true_flip_angle_fn      = [subj_label '_' sess_label '_acq-famp_run-1_TB1TFL_space-withinGRE-' kp{1} '.nii.gz'];
    true_flip_angle_json	= [subj_label '_' sess_label '_acq-famp_run-1_TB1TFL.json'];
    
    true_flip_angle         = load_nii_img_only( fullfile( ANTs_b1_apply_gre_dir, true_flip_angle_fn));
    b1_header               = jsondecode( fileread( fullfile( converted_b1_dir,true_flip_angle_json)));
    
    b1                      = true_flip_angle / 10 / b1_header.FlipAngle;
    
    mask_fn         = [subj_label '_' sess_label '_acq-' kp{1} '_' run_label '_brain_mask.nii.gz'];
    mask_filename   = fullfile(prot_SEPIA_dir, mask_fn);
    mask            = load_nii_img_only(mask_filename);
    
end
% 
% slice   = 106;
% img             = img(:,:,slice,:,:);
% unwrappedPhase  = unwrappedPhase(:,:,slice,:,:);
% mask            = mask(:,:,slice);
% b1              = b1(:,:,slice);
% totalField      = totalField(:,:,slice,:);

clearvars -except slice img mask b1 fa tr sepia_header unwrappedPhase totalField

% load(fullfile('/project/3015069.05/machine_learning_development/ANN_epgx_perceptron_round3/MLP_EPGX_leakyrelu_LeeANN4MWImodel_strategy2',...
%     'MLP_EPGX_leakyrelu_LeeANN4MWImodel_strategy2_epoch_100.mat'))
% 
% %%
% img = img .* exp(1i*unwrappedPhase);
% 
% mask_nonnan = min(min( ~isnan(img), [], 5), [],4);
% 
% mask = and(mask,mask_nonnan);
% 
% pini = unwrappedPhase - 2*pi*permute(totalField,[1 2 3 5 4]) .* permute(sepia_header{end}.TE(:),[2 3 4 1 5]);
% pini = polyfit3D_NthOrder( double(mean(pini(:,:,:,1,1:3), 5)), mask, 6);

% %% single-compartment
% s0 = zeros(size(img,1),size(img,2),size(img,3),size(img,5));
% r2s = zeros(size(img,1),size(img,2),size(img,3),size(img,5));
% for k = 1:size(img,5)
%     [r2s(:,:,:,k),~,s0(:,:,:,k)] = R2star_trapezoidal(abs(img(:,:,:,:,k)),sepia_header{end}.TE);
% end
% 
% r2s = mean(r2s,4);
% [t1,m0] = despot1_mapping(s0,fa,tr,mask,b1);
% 
% clear s0
% 
% mask_r2s = r2s >=5;
% mask = and(mask,mask_r2s);



%% define model
img = abs(img);

algorParam.iterations       = 6000;
algorParam.initialLearnRate = 0.1;      % Specify Training Options
algorParam.decayRate        = 0;
algorParam.display          = true;
algorParam.saveintermediate = true;

imgParam.te         = sepia_header{end}.TE; % in second
imgParam.tr         = tr;                   % in second
imgParam.fa         = fa;                   % in degree
imgParam.img        = img;
imgParam.mask       = mask;
imgParam.b1         = b1;
imgParam.output_dir         = fullfile(pwd,  'singlecompartment_r1r2s_realdata_3d');
imgParam.output_filename    = 'test';

fitRes = singlecompartment_r1r2s_gpu(algorParam, imgParam);

%% plot result
z = 106;
figure('Units','normalized','Position',[0 0 1 1]);
subplot(1,3,1);
imshow(gather(extractdata(parameters.r2s(:,:,z))),[0 40]);title('GPU R2*')

subplot(1,3,2);
imshow(gather(extractdata(parameters.m0(:,:,z)*scaleFactor)),[0 scaleFactor*10]);title('GPU M0')
    
subplot(1,3,3);
imshow(gather(extractdata(parameters.r1(:,:,z))),[0 3]);title('GPU R1')

% export_fig(fullfile(output_dir,'ann_gpu_parameter_set1'),'-png')


% %% plot result
% 
% for epoch =500:500:6000
%     load(fullfile(output_dir,['epoch_' num2str(epoch)]),'dloptimiser')
% figure('Units','normalized','Position',[0 0 1 1]);
% subplot(2,4,1);
% imshow(gather(extractdata(dloptimiser.parameters.r2sa)),[0 20]);title('GPU R2* IEW')
% subplot(2,4,2);
% imshow(gather(extractdata(dloptimiser.parameters.r2sb)),[0 150]);title('GPU R2* MW')
% subplot(2,4,3);
% imshow(gather(extractdata(dloptimiser.parameters.s0a))*scaleFactor,[0 scaleFactor*10]);title('GPU M0 IEW')
% subplot(2,4,4);
% imshow(gather(extractdata(dloptimiser.parameters.s0b))*scaleFactor,[0 scaleFactor*10*0.2]);title('GPU M0 MW')
% subplot(2,4,5);
% imshow(gather(extractdata(dloptimiser.parameters.r1a)),[0 3]);title('GPU R1 IEW')
% subplot(2,4,6);
% imshow(gather(extractdata(dloptimiser.parameters.kab)),[0 8]);title('GPU kIEWMW')
% subplot(2,4,7);
% imshow(gather(extractdata(dloptimiser.parameters.s0b ./ (dloptimiser.parameters.s0b + dloptimiser.parameters.s0a))),[0 0.3]);title('GPU MWF');colorbar
% 
% sgtitle(['Epoch ' num2str(epoch) ]);
% 
% export_fig(fullfile(output_dir,['2pool_R1R2s_gpu_epgx_epoch-' num2str(epoch)]),'-png')
% 
% end