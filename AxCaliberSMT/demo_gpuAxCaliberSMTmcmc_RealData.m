%% demo_gpuAxCaliberSMTmcmc_RealData.m
%
% This demo provides several examples on the ulitisation of gpuAxCaliberSMTmcmc.m 
% for parameter estimation with in vivo data
% 
% Kwok-Shing Chan 
% kchan2@mgh.harvard.edu
%
% Date created: 25 March 2024 
% Date modified:
%
%% add paths
addpath(genpath('/autofs/space/linen_001/users/kwokshing/tools/askadam')); % this path should be accessible to the group
clear;

%% I/O: Load data
data_dir = '/autofs/cluster/connectome2/Bay8_C2/bids/derivatives/processed_dwi/sub-010';

% Nb: # of unique b-value per little delta per big delta
% Nd: # of unique little delta
% ND: # of unique big delta

dwi     = niftiread(fullfile(data_dir, 'sub-010_dwi.nii.gz'));                      % full DWI data 
mask    = niftiread(fullfile(data_dir, 'sub-010_brain_mask.nii.gz'))>0;             % signal mask
bval    = readmatrix(fullfile(data_dir,'sub-010.bval'),         'FileType','text'); % 1x(Nb*Nd*ND) b-values, same length as the 4th dimension dwi
bvec    = readmatrix(fullfile(data_dir,'sub-010.bvec'),         'FileType','text'); % 3x(Nb*Nd*ND) gradient directions, 2nd dimension has the same length as the 4th dimension dwi
ldelta  = readmatrix(fullfile(data_dir,'sub-010.pulseWidth'),   'FileType','text'); % 1x(Nb*Nd*ND) little delta, same length as the 4th dimension dwi
BDELTA  = readmatrix(fullfile(data_dir,'sub-010.diffusionTime'),'FileType','text'); % 1x(Nb*Nd*ND) big delta, same length as the 4th dimension dwi

%% Algorithm parameters
bval        = bval/1e3; % s/mm2 to ms/um2
% fix parameters
D0          = 2;
Da_fixed    = 1.7;
DeL_fixed   = 1.7;
Dcsf        = 3;

% get unique b-values for each little delta and big delta
[bval_sorted,ldelta_sorted,BDELTA_sorted] = preparationDWI.unique_shell(bval,ldelta,BDELTA);

%% Usage #1: Basic default setting (same as Hong-Hsi's original implementation)
fitting             = [];
fitting.iteration   = 2e4;      % 2e4 for demo purpose. Original implementation used 2e5.
fitting.sampling    = 100;
fitting.method      = 'median';
fitting.start       = 'default';

% get the GPU device
g = gpuDevice;
% intiate MCMC object
smt_gpu                     = gpuAxCaliberSMTmcmc(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);
% MCMC estimation
[out,r,f,fcsf,DeR,noise]    = smt_gpu.estimate(dwi, mask, bval, bvec, ldelta, BDELTA, fitting);
% reset GPU memory
reset(g)

% Here is an example to convert posterior distribution into image space
r_dist = smt_gpu.distribution2image(out.r_dist,mask);

%% Usgae #2: You may use the internal save option to export the posterior distributions into disk
%  This may be useful when the #MCMC iterations is large to avoid memory issue
fitting                 = [];
fitting.iteration       = 2e4;
fitting.sampling        = 100;
fitting.method          = 'median';
fitting.start           = 'default';
fitting.output_filename = '/path/to/output_dir/filename.mat';   % provide your output info here

g = gpuDevice;
smt_gpu                   = gpuAxCaliberSMTmcmc(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);
[~,r,f,fcsf,DeR,noise]    = smt_gpu.estimate(dwi, mask, bval, bvec, ldelta, BDELTA, fitting);
reset(g)

% to load the distribution back to Workspace
load(fitting.output_filename);
r_dist = smt_gpu.distribution2image(out.r_dist,mask);

%% Usgae #3: If you perfer to use the spherical mean signal as the input it is also possible
% This might be useful if you already have the SMT signal computed and don't want to laod the entire DWI into memory
% You may use whatever method here, just make sure the DWI order in the 4-th dimension  must match the bval/ldelta/BDELTA order used in Line #77)
obj     = preparationDWI;
lmax    = 0;
dwi_smt = obj.get_Sl_all(dwi,bval,bvec,ldelta,BDELTA,lmax);

fitting                 = [];
fitting.iteration       = 2e4;
fitting.sampling        = 100;
fitting.method          = 'median';
fitting.start           = 'default';

g = gpuDevice;
smt_gpu                   = gpuAxCaliberSMTmcmc(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);
[~,r,f,fcsf,DeR,noise]    = smt_gpu.estimate(dwi_smt, mask, bval, bvec, ldelta, BDELTA, fitting);
reset(g)

%% Usgae #5: You might also define the starting points yourself
fitting                 = [];
fitting.iteration       = 2e4;
fitting.sampling        = 100;
fitting.method          = 'median';
fitting.start           = [3,0.5,0.20,0.7,0.02];

g = gpuDevice;
smt_gpu                   = gpuAxCaliberSMTmcmc(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);
[~,r,f,fcsf,DeR,noise]    = smt_gpu.estimate(dwi, mask, bval, bvec, ldelta, BDELTA, fitting);
reset(g)

%% Below list a few methods to reduce the number of outliers in the MCMC results
% The three options are not mutual exclusive so you may use all or some of the options together

%% Option 1: Use maximum likelihood to initiate estimation starting points
fitting             = [];
fitting.iteration   = 2e4;
fitting.sampling    = 100;
fitting.method      = 'median';
fitting.start       = 'likelihood';

g = gpuDevice;
smt_gpu                     = gpuAxCaliberSMTmcmc(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);
[out,r,f,fcsf,DeR,noise]    = smt_gpu.estimate(dwi, mask, bval, bvec, ldelta, BDELTA, fitting);
reset(g)

%% Option 2: Use narrower parameter range 
fitting             = [];
fitting.iteration   = 2e4;
fitting.sampling    = 100;
fitting.method      = 'median';
fitting.start       = 'likelihood';
fitting.boundary    = [ 0.1 5         ;   % radius, um
                          0 1         ;   % intra-cellular volume fraction
                          0 1         ;   % isotropic volume fraction
                       0.2 DeL_fixed  ;   % extra-cellular RD, um2/ms
                       0.01 1        ];

g = gpuDevice;
smt_gpu                     = gpuAxCaliberSMTmcmc(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);
[out,r,f,fcsf,DeR,noise]    = smt_gpu.estimate(dwi, mask, bval, bvec, ldelta, BDELTA, fitting);
reset(g)

%% Option 3: Run MCMC with multiple times at the same starting position, the final results will be the median/mean across all repetitions
t = tic;

fitting             = [];
fitting.repetition  = 5;
fitting.iteration   = 2e4;
fitting.sampling    = 100;
fitting.method      = 'median';
fitting.start       = 'default';

g = gpuDevice;
smt_gpu                     = gpuAxCaliberSMTmcmc(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);
[out,r,f,fcsf,DeR,noise]    = smt_gpu.estimate(dwi, mask, bval, bvec, ldelta, BDELTA, fitting);
reset(g)

toc(t)

% Here is an example to convert posterior distribution into image space
r_dist = smt_gpu.distribution2image(out.r_dist,mask);

%% Advanced Option 4: Run MCMC with multiple times, with different starting points, using multiple GPUs
% which may or may not speed up the process depending on the setup
% if you want to save the posterior distributions it would be better to run in serial (i.e. the option above)
% Note that the display messages will be scrumbled
t = tic;

random_bound = 0.2; % randomly pick +/-20% starting points

% check available GPU
availableGPUs = gpuDeviceCount("available");
if availableGPUs > 1 && isempty(gcp('nocreate'))
pool = parpool("Processes",availableGPUs);
end

repetition = 5;
% out     = repmat(struct(), repetition, 1 );
r       = zeros([size(mask) repetition]);
f       = zeros([size(mask) repetition]);
fcsf    = zeros([size(mask) repetition]);
DeR     = zeros([size(mask) repetition]);
noise   = zeros([size(mask) repetition]);

g = gpuDevice;

if availableGPUs > 1

parfor k = 1:repetition
fitting                 = [];
fitting.repetition      = 1;
fitting.iteration       = 2e4;
fitting.sampling        = 100;
fitting.method          = 'median';
fitting.start           = [3,0.5,0.20,0.7,0.02] .* (1-random_bound)+(random_bound*2)*rand(1,5);
% if you want the posterior distribution of each iteration, you need to save it in the disk space instead
% fitting.output_filename = strcat('/path/to/output_dir/filename_',num2str(k),'.mat');   
fitting.output_filename = [];

smt_gpu                     = gpuAxCaliberSMTmcmc(bval_sorted, ldelta_sorted, BDELTA_sorted, D0, Da_fixed, DeL_fixed, Dcsf);
[~,r(:,:,:,k),f(:,:,:,k),fcsf(:,:,:,k),DeR(:,:,:,k),noise(:,:,:,k)]    = smt_gpu.estimate(dwi, mask, bval, bvec, ldelta, BDELTA, fitting);

end
toc(t)

reset(g)
delete(pool)

r_final = median(r,4);

end