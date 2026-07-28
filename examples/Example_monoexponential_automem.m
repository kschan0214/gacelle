addpath(genpath('../../gacelle/'))
clear

%% generate some signal based on monoexponential decay
% reproducibility
seed = 5438973; rng(seed); gpurng(seed);

objGPU              = gpuR2starMapping(t);

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
S                     = objGPU.FWD(pars);

% realistic signal with certain SNR
noise   = mean(M0(:)) / SNR;        % estimate noise level
y       = S + noise*randn(size(S)); % add Gaussian noise

%% DEMO#1: askadam estimation default
objGPU              = gpuR2starMapping(t);
fitting             = [];
fitting.solver      = 'askadam';
fitting             = objGPU.check_set_default(fitting);
fitting.start       = 'default';

objGPU              = gpuR2starMapping(t);
out_adam            = objGPU.estimate(y, mask, fitting);

%% plot the estimation results
figure;
scatter(M0(mask>0),out_adam.final.(modelParams{1})(mask>0));refline(1);
xlabel('GT'); ylabel('S0')
nexttile;scatter(R2star(mask>0),pars0.(modelParams{2})(mask>0));hold on; scatter(R2star(mask>0),out_adam.final.(modelParams{2})(mask>0));refline(1)
xlabel('GT'); ylabel('R2*')
legend('Start','fitted')
figure; tiledlayout(2,3)
nexttile;imshow(M0(:,:,201).*mask(:,:,201),[0 2]);title('S0 GT')
nexttile;imshow(pars0.(modelParams{1})(:,:,201).*mask(:,:,201),[0 2]);title('S0 Start')
nexttile;imshow(out.final.(modelParams{1})(:,:,201),[0 2]);title('S0 Fitted')
nexttile;imshow(R2star(:,:,201).*mask(:,:,201),[10 60]);title('R2* GT')
nexttile;imshow(pars0.(modelParams{2})(:,:,201).*mask(:,:,201),[10 60]);title('R2* Start')
nexttile;imshow(out.final.(modelParams{2})(:,:,201),[10 60]);title('R2* Fitted')

