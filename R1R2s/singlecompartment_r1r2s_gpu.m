%% fitRes = singlecompartment_r1r2s_gpu(algorParam, imgParam)
%
% Input
% --------------
% 
%
% Output
% --------------
%
% Description: perform simultaneous single compartment R1-R2* fitting using
% Adam optimiser
%
% Kwok-shing Chan @ DCCN
% kwokshing.chan@donders.ru.nl
% Date created: 2 Nov 2021
% Date modified:
%
%
function fitRes = singlecompartment_r1r2s_gpu(algorParam, imgParam)

te      = imgParam.te;
tr      = imgParam.tr;
fa      = imgParam.fa;
img     = imgParam.img;
mask    = imgParam.mask;
b1      = imgParam.b1;

numEpochs           = algorParam.iterations;
initialLearnRate    = algorParam.initialLearnRate;
decayRate           = algorParam.decayRate;
isdisplay           = algorParam.display;
isSaveIntermediate  = algorParam.saveintermediate;

output_dir      = imgParam.output_dir;
output_filename = imgParam.output_filename;
isAutoSave      = imgParam.autoSave;

%% preprocess data
dims(1) = size(img,1);
dims(2) = size(img,2);
dims(3) = size(img,3);
nTE     = length(te);
nFA     = length(fa);

% vector-based, good for speed
te      = permute(te(:),[2 1]);
fa_map  = b1(mask>0) .* permute(deg2rad(fa(:)),[2 3 1]);
% % image-based 
% % te should be [1,1,1,nTE]
% te = permute(te(:),[2 3 4 1]);
% % fa_map should be [sx, sy, sz, 1, nFA]
% fa     = deg2rad(fa);   % convert from degree to radian
% fa_map = b1 .* permute(fa(:),[2 3 4 5 1]);  % incorporate B1


% normalise to [0,1]
scaleFactor = max(abs(img),[], 'all');
% scaleFactor = max(m0,[],'all');
img             = img / scaleFactor;

numMaskedVoxel  = numel(mask(mask>0));
% mask            = repmat(mask,[1 1 1 numel(te) length(fa)]);

%% define model
% get model parameters
parameters = initialise_model_singlecompartment_r1r2s_1d(numMaskedVoxel);

% clear cache before running everthing
accfun = dlaccelerate(@modelGradients_singlecompartment_r1r2s_1d);
clearCache(accfun)
% clear accfun
% accfun = dlaccelerate(@modelGradients_singlecompartment_r1r2s_1d);

% preprae data for gpu
img     = reshape(img,[prod(dims),nTE,nFA]);
img     = gpuArray( dlarray(img(mask(:)>0,:,:)));
te      = gpuArray( dlarray(te));
% mask    = gpuArray( dlarray(mask));
fa_map  = gpuArray( dlarray(fa_map));
tr      = gpuArray( dlarray(tr));

if isdisplay
    figure
    C = colororder;
    lineLoss = animatedline('Color',C(2,:));
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
    
    start = tic;
end

averageGrad     = [];
averageSqGrad   = [];
% optimisation
for epoch = 1:numEpochs
        
    % Evaluate the model gradients and loss using dlfeval and the modelGradients function.
    [gradients,loss] = dlfeval(accfun,parameters,img, te, fa_map, tr);

    % Update learning rate.
    learningRate = initialLearnRate / (1+decayRate*epoch);

    % Update the network parameters using the adamupdate function.
    [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
        averageSqGrad,epoch,learningRate);

        
    % Plot training progress.
    loss = double(gather(extractdata(loss)));
    if isdisplay
        addpoints(lineLoss,epoch, loss);
    
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
        drawnow
    end
    
    if mod(epoch,100) == 0 && isdisplay
        % print intermediate
        rse     = sqrt(sum(gather(extractdata( ((abs(img - model_singlecompartment_r1r2s(parameters,te,fa_map,tr))).^2))),'all','omitnan'));
        NRSE    =  rse / gather(extractdata(sqrt(sum(abs(img).^2,'all'))));
        fprintf('#Epoch: %i, NRSE: %f \n',epoch,NRSE);
    end

    % save intermediate results
    if mod(epoch,250) == 0 && isSaveIntermediate
        dloptimiser.parameters    = parameters;
        dloptimiser.epoch         = epoch;
        save(fullfile(output_dir,[output_filename '_epoch_' num2str(epoch)]),'dloptimiser');
        export_fig(fullfile(output_dir,[output_filename '_loss_' num2str(epoch)]),'-png');
    end
end

rse     = sqrt(sum(gather(extractdata( ((abs(img - model_singlecompartment_r1r2s(parameters,te,fa_map,tr))).^2))),'all','omitnan'));
NRSE    =  rse / gather(extractdata(sqrt(sum(abs(img).^2,'all'))));

fitRes.m0   = zeros(dims,'single'); fitRes.m0(mask>0)    = parameters.m0 * scaleFactor;
fitRes.r1   = zeros(dims,'single'); fitRes.r1(mask>0)    = parameters.r1;
fitRes.r2s  = zeros(dims,'single'); fitRes.r2s(mask>0)   = parameters.r2s *10;

fitRes.loss = loss;
fitRes.nrse = NRSE;

if isAutoSave
    save(fullfile(output_dir,output_filename),'fitRes');
end

end