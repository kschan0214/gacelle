addpath('/autofs/space/linen_001/users/kwokshing/tools/sepia/sepia_master/')
sepia_addpath
 
%% create output directory

input_dir = '/homes/1/kc020/Desktop/Bay8_C2/bids/derivatives/processed_dwi/sub-001/';
subj_label = 'sub-001';

% if ~exist(output_dir,'dir')
%     mkdir(output_dir)
% end

%%
D_txt       = strcat(subj_label,'.diffusionTime');
bvals_txt   = strcat(subj_label,'.bval');
bvecs_txt   = strcat(subj_label,'.bvec');
dwi_nii     = strcat(subj_label,'_dwi.nii.gz');

D       = readtable(fullfile(input_dir,D_txt),"FileType","text");     D = D{1,:};
bval    = readtable(fullfile(input_dir,bvals_txt),"FileType","text"); bval = bval{:,:};
bvec    = readtable(fullfile(input_dir,bvecs_txt),"FileType","text"); bvec = bvec{:,:};
nii     = load_untouch_nii(fullfile(input_dir,dwi_nii));
dwi     = nii.img;
mask_nii = fullfile(input_dir, strcat(subj_label,'_brain_mask.nii.gz'));
nii     = load_untouch_nii(mask_nii);
mask    = nii.img;
D_unique = unique(D(D~=0));

for kd = 1%:numel(D_unique)

% use only one D
idx = find(D == D_unique(kd));

D_tmp   = D(idx);
bval_tmp= bval(idx);
bvec_tmp= bvec(:,idx);
dwi_tmp = dwi(:,:,:,idx);
te_tmp = zeros(size(bval_tmp));
ldelta_tmp = ones(size(bval_tmp))*6;

DWIutils = DWIutility;
bval_tmp = DWIutils.rectify_bval_v2(bval_tmp,0.01);
[bval_sorted,ldelta_sorted,BDELTA_sorted,~] = DWIutils.unique_shell_keepb0(bval_tmp,ldelta_tmp,D_tmp);

Ds = 3;
objGPU                      = gpuSANDI(bval_sorted/1e3,ldelta_sorted,BDELTA_sorted,Ds);
fitting                     = [];
fitting.solver              = 'mcmc';
fitting.algorithm           = 'ensemble';
fitting.Nwalker             = 50;
fitting.StepSize            = 2;
fitting.iteration           = 1e4;
fitting.thinning            = 10;        % Sample every 10 iteration
fitting.metric              = {'median','iqr'};
fitting.burnin              = 0.1;       % 10% burn-in
fitting.start               = 'default';

extraData                   = [];
extraData.bval              = bval_tmp/1e3;
extraData.bvec              = bvec_tmp;
extraData.BDELTA            = D_tmp;
extraData.ldelta            = ldelta_tmp;

out   = objGPU.estimate(dwi_tmp, mask, fitting,extraData);


end
