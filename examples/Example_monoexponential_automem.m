%% Example_monoexponential_automem.m
%
% Demonstrates GACELLE's automatic GPU memory management
% (fitting.autoMemManage) via gpuR2starMapping on a monoexponential R2*
% decay: a 401x401x401 volume (~65M voxels) is deliberately chosen to be
% far too large to fit on typical GPU VRAM in one pass. A spherical mask,
% simulated noisy signal, and default askadam fitting settings are used;
% the only thing being exercised here is that estimate() transparently
% segments the volume into GPU-sized chunks and stitches the result back
% together, rather than needing the user to do that manually.
%
% Kwok-Shing Chan
% kchan2@mgh.harvard.edu
%
% Date created: 5 August 2026
% Date modified:
%
addpath('../../gacelle/');addpath_gacelle;
clear

%% generate some signal based on monoexponential decay
% reproducibility
seed = 5438973; rng(seed); gpurng(seed);

% set up estimation parameters; must be the same as in FWD function
modelParams = {'M0','R2star'};

% define number of voxels and SNR
Nx      = 401;
Ny      = 401;
Nz      = 401;
SNR     = 100;
% let's create a spherical mask
mask        = strel('sphere',(Nx-1)/2);mask = mask.Neighborhood;
t           = linspace(0,40e-3,15);
% GT
M0          = 1 + randn(Nx,Ny,Nz)*0.3;
R2star      = 30 + 5*randn(Nx,Ny,Nz);
% forward signal generation
pars.(modelParams{1}) = M0;
pars.(modelParams{2}) = R2star;
% S now is a 4D matrix
objGPU              = gpuR2starMapping(t);
S                     = objGPU.FWD(pars);

% realistic signal with certain SNR
noise   = mean(M0(:)) / SNR;        % estimate noise level
y       = S + noise*randn(size(S)); % add Gaussian noise

%% DEMO#1: askadam estimation default
% fitting.autoMemManage defaults to on (see
% docs/advanced/automatic_memory_management.rst) - estimate() will probe
% available GPU memory and, if the full [Nx,Ny,Nz,Nt] volume doesn't fit,
% automatically split it into density-balanced, halo-padded chunks
% (utils.find_optimal_segment_3D) processed sequentially. No extra input
% is required from the caller for this to happen.
objGPU              = gpuR2starMapping(t);
fitting             = [];
fitting.solver      = 'askadam';
fitting             = objGPU.check_set_default(fitting);
fitting.start       = 'default';

objGPU              = gpuR2starMapping(t);
out_adam            = objGPU.estimate(y, mask, fitting);

%% plot the estimation results
figure;
binscatter(pars.(modelParams{1})(mask>0),out_adam.final.(modelParams{1})(mask>0));h = refline(1); h.Color = 'k';
xlabel('GT'); ylabel('S0')
nexttile; binscatter (pars.(modelParams{2})(mask>0),out_adam.final.(modelParams{2})(mask>0));h = refline(1); h.Color = 'k';
xlabel('GT'); ylabel('R2*')
legend('Start','fitted')
figure; tiledlayout(2,3)
nexttile;imshow(M0(:,:,201).*mask(:,:,201),[0 2]);title('S0 GT')
nexttile;imshow(pars.(modelParams{1})(:,:,201).*mask(:,:,201),[0 2]);title('S0 Start')
nexttile;imshow(out_adam.final.(modelParams{1})(:,:,201),[0 2]);title('S0 Fitted')
nexttile;imshow(R2star(:,:,201).*mask(:,:,201),[10 60]);title('R2* GT')
nexttile;imshow(pars.(modelParams{2})(:,:,201).*mask(:,:,201),[10 60]);title('R2* Start')
nexttile;imshow(out_adam.final.(modelParams{2})(:,:,201),[10 60]);title('R2* Fitted')

