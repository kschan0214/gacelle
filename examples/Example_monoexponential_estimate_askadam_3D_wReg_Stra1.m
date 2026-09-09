%% Example_monoexponential_estimate_askadam_3D_wReg_Stra1.m
%
% Demonstrates askadam.optimisation's spatial-regularisation options on a
% 3D monoexponential (S0, R2*) fit, using the "Strategy 1" forward
% function (Example_monoexponential_FWD_askadam_3D_Strategy1.m -
% straightforward but not memory-optimised; see the _wReg_Stra2 companion
% script for the more efficient Strategy 2 forward function). Runs the
% same fit 4 ways to show every combination of:
%   - built-in spatial total-variation regularisation
%     (@spatial_total_variation, wired in via fitting.regmap/.lambda/
%     .TVmode/.voxelSize) vs. a user-supplied regularisation function
%     (passed explicitly alongside the forward model, for when the
%     built-in TV penalty isn't what you need);
%   - regularising R2* only vs. both S0 and R2*.
%
% Kwok-Shing Chan
% Date created: 5 August 2026
% Date modified:
%
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
S                     = Example_monoexponential_FWD_askadam_3D_Strategy1(pars,t);

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
% built-in spatial TV regularisation: applied only to R2star (regmap),
% weighted by lambda, using 2D (in-plane) TV mode
fitting.regmap              = fitting.modelParams(2);
fitting.lambda              = {0.002};
fitting.TVmode              = '2D';
fitting.voxelSize           = [1,1,1];
fitting.isOptimiseMemory    = false;

% define your forward model
modelFWD    = @Example_monoexponential_FWD_askadam_3D_Strategy1;

% equal weights
weights = [];

askadam_obj = askadam;
out_builtin = askadam_obj.optimisation(y,mask,weights,pars0,fitting,modelFWD,t);

%% using user defined regularisation function on R2*
% Same fit as above, but the built-in TV penalty is replaced with an
% explicit user-supplied regularisation function (here,
% @spatial_total_variation itself, called directly rather than through
% fitting.regmap/.lambda) to show the general pattern for plugging in a
% custom regulariser: userFcn/userInput each pack the forward model and
% the regularisation function (and their respective extra inputs) as a
% {forward; reg} cell pair.
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
fitting.isOptimiseMemory    = false;

% define your forward model
modelFWD    = @Example_monoexponential_FWD_askadam_3D_Strategy1;
regFcn      = @spatial_total_variation;
userFcn     = {modelFWD; regFcn};       % Position #1: forward model function; Position #2: regularisation function

% specify your model input
modelInput  = {t};  % following the same order as specified in the forward function, except the first input
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
% compares the built-in-regularisation fit against the user-function fit
% (they should match, since @spatial_total_variation is what the
% built-in path calls internally) alongside GT and the random start
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
% Same as the first demo above, but fitting.regmap now lists both S0 and
% R2star, each with its own lambda weight (0.15 and 0.002 respectively -
% note S0's much larger weight, since it varies on an O(1) scale vs.
% R2*'s O(10-100) scale).
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
fitting.isOptimiseMemory    = false;

% define your forward model
modelFWD    = @Example_monoexponential_FWD_askadam_3D_Strategy1;

% equal weights
weights = [];

askadam_obj = askadam;
out_builtin = askadam_obj.optimisation(y,mask,weights,pars0,fitting,modelFWD,t);

%%  using user defined regularisation function on both parameters
% User-function equivalent of the block above - regmap/lambda for both
% parameters are passed through regInput instead of fitting.regmap/.lambda.
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
fitting.isOptimiseMemory    = false;

% define your forward model
modelFWD    = @Example_monoexponential_FWD_askadam_3D_Strategy1;
regFcn      = @spatial_total_variation;
userFcn     = {modelFWD; regFcn};       % Position #1: forward model function; Position #2: regularisation function

% specify your model input
modelInput  = {t};  % following the same order as specified in the forward function, except the first input
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
