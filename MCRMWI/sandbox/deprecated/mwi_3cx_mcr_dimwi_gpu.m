function fitRes = mwi_3cx_mcr_dimwi_gpu(algorParam, imgParam)
%% preprocess data
te          = imgParam.te;
tr          = imgParam.tr;
fa          = imgParam.fa;
img         = imgParam.img;
mask        = imgParam.mask;
b1          = imgParam.b1;
totalField  = imgParam.totalField;
pini        = imgParam.pini;
r1mw        = 1./imgParam.t1mw;
rho_mw      = imgParam.rho_mw;

% Specify Training Options
numEpochs           = algorParam.iterations;
initialLearnRate    = algorParam.initialLearnRate;
decayRate           = algorParam.decayRate;
isdisplay           = algorParam.display;
isSaveIntermediate  = algorParam.saveintermediate;

output_dir      = imgParam.output_dir;
output_filename = imgParam.output_filename;
isAutoSave      = imgParam.autoSave;

% ANN for EPG-X
addpath('/project/3015069.05/machine_learning_development/ANN_epgx_perceptron_round3');
load(fullfile('/project/3015069.05/machine_learning_development/ANN_epgx_perceptron_round3/MLP_EPGX_leakyrelu_LeeANN4MWImodel_strategy2',...
    'MLP_EPGX_leakyrelu_LeeANN4MWImodel_strategy2_epoch_100.mat'))
dlnet.alpha = 0.01;

%% preprocess data
dims(1) = size(img,1);
dims(2) = size(img,2);
dims(3) = size(img,3);
nTE     = length(te);
nFA     = length(fa);

% vector-based, good for speed
te      = permute(te(:),[2 1]);
fa_map  = b1(:) .* permute(deg2rad(fa(:)),[2 3 1]);
% fa_map  = b1(mask>0) .* permute(deg2rad(fa(:)),[2 3 1]);
% % image-based operation, good for regularisation
% % te should be [1,1,1,nTE]
% te = permute(te(:),[2 3 4 1]);
% % fa_map should be [sx, sy, sz, 1, nFA]
% fa_map = b1 .* permute(deg2rad(fa(:)),[2 3 4 5 1]);


% normalise to [0,1]
scaleFactor = max(abs(img),[], 'all');
img = img / scaleFactor;

% reshape matrix
img         = reshape(img,[prod(dims),nTE,nFA]);
% img         = img(mask(:)>0,:,:);
totalField  = reshape(totalField,[prod(dims),1,nFA]);
% totalField  = totalField(mask(:)>0,:,:);
% pini        = pini(mask(:)>0);
pini        = pini(:);
mask = repmat(mask,[1 1 1 numel(te) length(fa)]);
mask         = reshape(mask,[prod(dims),nTE,nFA]);

%% define model
% get model parameters
parameters = initialise_model_mcr_1d(abs(img),totalField,pini);
parameter_scale.r2siw   = 10;
parameter_scale.r2sew   = 20;
parameter_scale.r2smw   = 100;
parameter_scale.freq_mw = 10;
% parameters = initialise_model_mcr_3d_normalise(max(max(abs(img),[],4),[],5),totalField,pini);

accfun = dlaccelerate(@modelGradients_mcr_1d);
% accfun = dlaccelerate(@modelGradients_mcr_3d_weighted_1stecho_normalised);
clearCache(accfun)

% derive weights, weighted by 1echo
weights = abs(img) ./ abs(img(:,1,:));
% weights = abs(img) ./ abs(img(:,:,:,1,:));
weights(isnan(weights)) = 0;
weights(isinf(weights)) = 0;
weights(weights>1) = 1;

% weighted
img = img.*weights;
% img = cat(1,real(img(:)),imag(img(:))).'; % concatenate real and imaginary

% preprae data for gpu
img     = gpuArray( dlarray(img));
te      = gpuArray( dlarray(te));
fa_map  = gpuArray( dlarray(fa_map));
tr      = gpuArray( dlarray(tr));

figure
C = colororder;
lineLoss = animatedline('Color',C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

start = tic;

averageGrad     = [];
averageSqGrad   = [];
iteration = 0;
% optimisation
for epoch = 1:numEpochs
    iteration = iteration + 1;
        
    % Evaluate the model gradients and loss using dlfeval and the
    % modelGradients function.
    [gradients,loss] = dlfeval(accfun,parameters,img, te, fa_map, tr, weights, r1mw,rho_mw, dlnet,parameter_scale,mask(:));
%     [gradients,loss] = dlfeval(accfun,parameters,img, te, fa_map, tr, r1mw, mask, weights, dlnet);

    % Update learning rate.
    learningRate = initialLearnRate / (1+decayRate*iteration);

    % Update the network parameters using the adamupdate function.
    [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
        averageSqGrad,iteration,learningRate);

%     parameters = validate_update_parameters_ann(parameters);
        
    % Plot training progress.
    loss = double(gather(extractdata(loss)));
    addpoints(lineLoss,iteration, loss);

    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
    drawnow
    
%     if mod(epoch,100) == 0
%         % print intermediate
%         rse     = sqrt(sum(gather(extractdata( ((abs(img - model_mcr_ann(parameters,te,fa_map,tr,r1mw,mask(:,:,:,1,:),dlnet))).^2).*mask)),'all','omitnan'));
%         NRSE    =  rse / gather(extractdata(sqrt(sum(abs(img).^2.*mask,'all'))));
%         fprintf('#Epoch: %i, NRSE: %f \n',epoch,NRSE);
% 
% %         % save intermediate results
%         dloptimiser.parameters    = parameters;
%         dloptimiser.epoch         = epoch;
%         save(fullfile(output_dir,['epoch_' num2str(epoch)]),'dloptimiser');
%         export_fig(fullfile(output_dir,['epgx_loss_' num2str(epoch)]),'-png');
%     end
end

end

% export_fig('r1r2s_epgx_loss','-png');
