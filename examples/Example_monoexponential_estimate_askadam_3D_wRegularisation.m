addpath(genpath('../../gacelle/'))
clear

%% generate some signal based on monoexponential decay
% reproducibility
seed = 5438973; rng(seed); gpurng(seed);

% set up estimation parameters; must be the same as in FWD function
modelParams = {'S0','R2star'};

% define number of voxels and SNR
Nx      = 21;
Ny      = 21;
Nz      = 21;
SNR     = 100;
% let's create a spherical mask
mask        = strel('sphere',10);mask = mask.Neighborhood;
t           = linspace(0,40e-3,15); 
% GT
S0          = 1 + randn(Nx,Ny,Nz)*0.3;
R2star      = 30 + 5*randn(Nx,Ny,Nz);
% forward signal generation
pars.(modelParams{1}) = S0; 
pars.(modelParams{2}) = R2star;
% S now is a 4D matrix
S                     = Example_monoexponential_FWD_askadam_3D_Strategy1(pars,t,mask);

% realistic signal with certain SNR
noise   = mean(S0(:)) / SNR;        % estimate noise level
y       = S + noise*randn(size(S)); % add Gaussian noise

%% using built-in spatial TV regularisation function on R2*
% set up starting point
pars0.(modelParams{1}) = 1 + randn(Nx,Ny,Nz)*0.5;  % S0
pars0.(modelParams{2}) = 20 + 10*randn(Nx,Ny,Nz);   % R2*

% set up fitting algorithm
fitting                     = [];
% define model parameter name and fitting boundary
fitting.modelParams         = {'S0','R2star'}; % modelParams;
fitting.lb                  = [0, 0];   % lower bound 
fitting.ub                  = [2, 50];  % upper bound
% Estimation algorithm setting
fitting.iteration           = 4000;
fitting.initialLearnRate    = 0.001;
fitting.lossFunction        = 'l1';
fitting.tol                 = 1e-4;
fitting.convergenceValue    = 1e-8;
fitting.convergenceWindow   = 20;
fitting.isDisplay           = false;
fitting.regmap              = fitting.modelParams(2);
fitting.lambda              = {0.002};
fitting.TVmode              = '2D';
fitting.voxelSize           = [1,1,1];

% define your forward model
modelFWD    = @Example_monoexponential_FWD_askadam_3D_Strategy1;

% equal weights
weights = [];

askadam_obj = askadam;
out_builtin = askadam_obj.optimisation(y,mask,weights,pars0,fitting,modelFWD,t,mask);

%% using user defined regularisation function on R2*
% set up fitting algorithm
fitting                     = [];
% define model parameter name and fitting boundary
fitting.modelParams         = {'S0','R2star'}; % modelParams;
fitting.lb                  = [0, 0];   % lower bound 
fitting.ub                  = [2, 50];  % upper bound
% Estimation algorithm setting
fitting.iteration           = 4000;
fitting.initialLearnRate    = 0.001;
fitting.lossFunction        = 'l1';
fitting.tol                 = 1e-4;
fitting.convergenceValue    = 1e-8;
fitting.convergenceWindow   = 20;
fitting.isDisplay           = false;

% define your forward model
modelFWD    = @Example_monoexponential_FWD_askadam_3D_Strategy1;
regFcn      = @spatial_total_variation;
userFcn     = {modelFWD; regFcn};       % Position #1: forward model function; Position #2: regularisation function

% specify your model input
modelInput  = {t,mask};  % following the same order as specified in the forward function, except the first input
regmap      = fitting.modelParams(2);
lambda      = {0.002};
TVmode      = '2D';
voxelSize   = [1,1,1];
regInput    = {mask, lambda, regmap, TVmode, voxelSize};
userInput   = {modelInput;regInput};    % Position #1: forward model extra input; Position #2: regularisation extra input

% equal weights
weights = [];

askadam_obj = askadam;
out_user    = askadam_obj.optimisation(y,mask,weights,pars0,fitting,userFcn,userInput);

%% plot the estimation results
figure;
nexttile;scatter(S0(mask>0),pars0.(modelParams{1})(mask>0));hold on; scatter(S0(mask>0),out_builtin.final.S0(mask>0));refline(1);
xlabel('GT'); ylabel('S0 built-in')
nexttile;scatter(S0(mask>0),pars0.(modelParams{1})(mask>0));hold on; scatter(S0(mask>0),out_user.final.S0(mask>0));refline(1);
xlabel('GT'); ylabel('S0 user')
nexttile;scatter(R2star(mask>0),pars0.(modelParams{2})(mask>0));hold on; scatter(R2star(mask>0),out_builtin.final.R2star(mask>0));refline(1)
xlabel('GT'); ylabel('R2* built-in')
nexttile;scatter(R2star(mask>0),pars0.(modelParams{2})(mask>0));hold on; scatter(R2star(mask>0),out_user.final.R2star(mask>0));refline(1)
xlabel('GT'); ylabel('R2* user')
legend('Start','fitted')
figure; tiledlayout(2,4)
nexttile;imshow(S0(:,:,11).*mask(:,:,11),[0 2]);title('S0 GT')
nexttile;imshow(pars0.(modelParams{1})(:,:,11).*mask(:,:,11),[0 2]);title('S0 Start')
nexttile;imshow(out_builtin.final.S0(:,:,11),[0 2]);title('S0 Fitted built-in')
nexttile;imshow(out_user.final.S0(:,:,11),[0 2]);title('S0 Fitted user')
nexttile;imshow(R2star(:,:,11).*mask(:,:,11),[10 60]);title('R2* GT')
nexttile;imshow(pars0.(modelParams{2})(:,:,11).*mask(:,:,11),[10 60]);title('R2* Start')
nexttile;imshow(out_builtin.final.R2star(:,:,11),[10 60]);title('R2* Fitted built-in')
nexttile;imshow(out_user.final.R2star(:,:,11),[10 60]);title('R2* Fitted user')

%% using built-in spatial TV regularisation function on both parameters
% set up starting point
pars0.(modelParams{1}) = 1 + randn(Nx,Ny,Nz)*0.5;  % S0
pars0.(modelParams{2}) = 20 + 10*randn(Nx,Ny,Nz);   % R2*

% set up fitting algorithm
fitting                     = [];
% define model parameter name and fitting boundary
fitting.modelParams         = {'S0','R2star'}; % modelParams;
fitting.lb                  = [0, 0];   % lower bound 
fitting.ub                  = [2, 50];  % upper bound
% Estimation algorithm setting
fitting.iteration           = 4000;
fitting.initialLearnRate    = 0.001;
fitting.lossFunction        = 'l1';
fitting.tol                 = 1e-4;
fitting.convergenceValue    = 1e-8;
fitting.convergenceWindow   = 20;
fitting.isDisplay           = false;
fitting.regmap              = fitting.modelParams;
fitting.lambda              = {0.15,0.002};
fitting.TVmode              = '2D';
fitting.voxelSize           = [1,1,1];

% define your forward model
modelFWD    = @Example_monoexponential_FWD_askadam_3D_Strategy1;

% equal weights
weights = [];

askadam_obj = askadam;
out_builtin = askadam_obj.optimisation(y,mask,weights,pars0,fitting,modelFWD,t,mask);

%%  using user defined regularisation function on both parameters
% set up fitting algorithm
fitting                     = [];
% define model parameter name and fitting boundary
fitting.modelParams         = {'S0','R2star'}; % modelParams;
fitting.lb                  = [0, 0];   % lower bound 
fitting.ub                  = [2, 50];  % upper bound
% Estimation algorithm setting
fitting.iteration           = 4000;
fitting.initialLearnRate    = 0.001;
fitting.lossFunction        = 'l1';
fitting.tol                 = 1e-4;
fitting.convergenceValue    = 1e-8;
fitting.convergenceWindow   = 20;
fitting.isDisplay           = false;

% define your forward model
modelFWD    = @Example_monoexponential_FWD_askadam_3D_Strategy1;
regFcn      = @spatial_total_variation;
userFcn     = {modelFWD; regFcn};       % Position #1: forward model function; Position #2: regularisation function

% specify your model input
modelInput  = {t,mask};  % following the same order as specified in the forward function, except the first input
regmap      = fitting.modelParams;
lambda      = {0.15,0.002};
TVmode      = '2D';
voxelSize   = [1,1,1];
regInput    = {mask, lambda, regmap, TVmode, voxelSize};
userInput   = {modelInput;regInput};    % Position #1: forward model extra input; Position #2: regularisation extra input

% equal weights
weights = [];

askadam_obj = askadam;
out_user    = askadam_obj.optimisation(y,mask,weights,pars0,fitting,userFcn,userInput);

%% plot the estimation results
figure;
nexttile;scatter(S0(mask>0),pars0.(modelParams{1})(mask>0));hold on; scatter(S0(mask>0),out_builtin.final.S0(mask>0));refline(1);
xlabel('GT'); ylabel('S0 built-in')
nexttile;scatter(S0(mask>0),pars0.(modelParams{1})(mask>0));hold on; scatter(S0(mask>0),out_user.final.S0(mask>0));refline(1);
xlabel('GT'); ylabel('S0 user')
nexttile;scatter(R2star(mask>0),pars0.(modelParams{2})(mask>0));hold on; scatter(R2star(mask>0),out_builtin.final.R2star(mask>0));refline(1)
xlabel('GT'); ylabel('R2* built-in')
nexttile;scatter(R2star(mask>0),pars0.(modelParams{2})(mask>0));hold on; scatter(R2star(mask>0),out_user.final.R2star(mask>0));refline(1)
xlabel('GT'); ylabel('R2* user')
legend('Start','fitted')
figure; tiledlayout(2,4)
nexttile;imshow(S0(:,:,11).*mask(:,:,11),[0 2]);title('S0 GT')
nexttile;imshow(pars0.(modelParams{1})(:,:,11).*mask(:,:,11),[0 2]);title('S0 Start')
nexttile;imshow(out_builtin.final.S0(:,:,11),[0 2]);title('S0 Fitted built-in')
nexttile;imshow(out_user.final.S0(:,:,11),[0 2]);title('S0 Fitted user')
nexttile;imshow(R2star(:,:,11).*mask(:,:,11),[10 60]);title('R2* GT')
nexttile;imshow(pars0.(modelParams{2})(:,:,11).*mask(:,:,11),[10 60]);title('R2* Start')
nexttile;imshow(out_builtin.final.R2star(:,:,11),[10 60]);title('R2* Fitted built-in')
nexttile;imshow(out_user.final.R2star(:,:,11),[10 60]);title('R2* Fitted user')
