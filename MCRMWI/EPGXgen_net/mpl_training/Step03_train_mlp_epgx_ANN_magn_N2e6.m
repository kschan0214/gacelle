%% Step03_train_mlp_epgx_ANN_magn_N2e6.m
%
% Generate EPG-X dictionary for MLP training
%
% using leaky relu
%
% Kwok-Shing Chan
% Date created: & August 2026
% 
clear

%% load data for training

epgx_dictionary_dir = fullfile(pwd,'MCRMWI_EPGX_dictionary_20240927_rfphase50_t1mylin234/');
output_prefix       = 'MCRMWI_MLP_EPGX_RFphase50_T1M234_magn';
output_dir          = fullfile(pwd,output_prefix);
if ~exist(output_dir,'dir'); mkdir(output_dir); end

% data preprocessing
Nfa         = 40;
Nbatch      = 40;
T1Myelin    = 234e-3;

% load dictionary data
% input_parameter - 1st dim: numSample; 2nd dim:fitting parameters
% Order of fitting parameter: fx-T1f-k-TR-T2iew
% Order of output parameter: ss_iew-ss_mw
input_parameter_all     = zeros(50000*Nbatch,5);
output_parameter_all    = zeros(50000*Nbatch,Nfa,2);
tic
for j = 1:Nbatch
    
    load(fullfile(epgx_dictionary_dir,['EPGX_steadystate_batch-' num2str(j) '.mat']));
    
    input_parameter_all((j-1)*50000+1:50000*j,:)      = input_parameter;
    output_parameter_all((j-1)*50000+1:50000*j,:,:)   = output_parameter;

end
input_parameter     = single(input_parameter_all);
output_parameter    = single(output_parameter_all);
toc

clearvars input_parameter_all output_parameter_all

%% create data structure for ANN training

% feature processing
nfeatures = 11;

fa  = linspace(1,90,Nfa);
% features - dim1: nsample; dim2: nFA; dim3: nfeatures 
features    = zeros(size(input_parameter,1),Nfa,nfeatures, 'single');
S_fw_BM     = zeros(size(input_parameter,1),Nfa);
S_m_BM      = zeros(size(input_parameter,1),Nfa);
for kfa = 1:numel(fa)
features(:,kfa,:) = feature_preprocess_MCRMWI_MLP_EPGX_leakyrelu(input_parameter(:,1),input_parameter(:,2),...
                                              input_parameter(:,3),deg2rad(fa(kfa)),input_parameter(:,4),input_parameter(:,5),T1Myelin).';

[S_m, S_fw]    = gpuMCRMWI.model_BM_2T1_analytical(input_parameter(:,4),deg2rad(fa(kfa)),input_parameter(:,1),1./input_parameter(:,2),1/T1Myelin,input_parameter(:,3));

S_fw_BM(:,kfa)  = S_fw;
S_m_BM(:,kfa)   = S_m;
end

features = reshape(features,[size(features,1) size(features,2)*size(features,3)]);

% create response variable
% 20211021 Just use the magnitude
% magnitude of IEW, magnitude of Mylin and phase of IEW
Res1        = abs(output_parameter(:,:,1))-S_fw_BM;    % difference between magnitude w/ and w/o EPGX FW
Res2        = abs(output_parameter(:,:,2))-S_m_BM;     % difference between magnitude w/ and w/o EPGX Myelin SS
response    = cat(2,Res1,Res2);

% create datastore
inds   = arrayDatastore(features.', 'IterationDimension',2);
outds  = arrayDatastore(response.', 'IterationDimension',2);

clear input_parameter output_parameter features response S_fw_BM S_m_BM Res1 Res2

ds = combine(inds,outds);

%% define DL model

% Specify the number of layers and the number of neurons for each layer.
numLayers   = [];
numNeurons  = [160 240 320 360 480 520 600]/8; % only hidden layer neurons
numInputs   = nfeatures; % size(feature,2);
numOutputs  = 2; % size(response,2);

% get model parameters
parameters = create_mlp(numLayers,numNeurons,numInputs,numOutputs);

disp(fullfile(output_dir,output_prefix))

%% Specify Training Options

Nepochs                 = 100;
executionEnvironment    = "gpu";

% hyper-parameters
% adaptive learning rate
initialLearnRate    = 0.01;
decayRate           = 0.0001;
alpha               = 0.01;     % leaky Relu scale factor

% adaptive batch size
miniBatchSize_step = [2^7 2^8 2^9 2^10 2^11];
miniBatchSize_counter = 0;

% 1st derivative of steady-state curve
initialRegularisation   = 100;
decayRateRegularisation = 0.1;
% adam parameter
averageGrad     = [];
averageSqGrad   = [];

if exist('accfun','var')
    clearCache(accfun)
    clear accfun
end
accfun = dlaccelerate(@modelGradients_mlp_epgx_ANN_corr_L1);

figure
C = colororder;
lineLoss = animatedline('Color',C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on
drawnow

% training loop
start       = tic;
iteration   = 0;
scounter    = 1;
for epoch = 1:Nepochs
    
    % every 10 epochs increase doubles batch size util 2048
    if mod(epoch,10) == 1
        if miniBatchSize_counter < length(miniBatchSize_step)

            miniBatchSize_counter   = miniBatchSize_counter + 1;
            miniBatchSize           = miniBatchSize_step(miniBatchSize_counter);
            % update batch size
            mbq = minibatchqueue(ds, ...
            'MiniBatchSize',miniBatchSize, ...
            'MiniBatchFormat','CB', ...
            'OutputEnvironment',executionEnvironment);
        end
        
    end

    reset(mbq);
    
    % update lamda for every batch
    lambda_curr = initialRegularisation / (1+decayRateRegularisation*(epoch-1));

    while hasdata(mbq)
        iteration = iteration + 1;
        
        % get the input and output from minibatch
        % dlin  - dim1: FA-features; dim2: sample
        % dlout - dim1: FA-pool;  dim2: sample
        [dlin, dlout] = next(mbq);

        % data reordering
        curr_batch_size = size(dlin,2);
        % dlin  - dim1: features; dim2: sample; dim3: FA
        % dlout - dim1: pool; dim2: sample; dim3: FA
        dlin        = permute(reshape(dlin, [Nfa,nfeatures,curr_batch_size]),[2 3 1]);
        dlout       = permute(reshape(dlout,[Nfa,numOutputs,curr_batch_size]), [2 3 1]);
        
        % gradient along FA
        % dldoutdfa - dim1: pool; dim2: sample-FA
        dldoutdfa   = dlout(:,:,2:end) - dlout(:,:,1:end-1);
        % dldoutdfa   = cat(1,dlout(1:2,:,2:end)+dlin(12:13,:,2:end) - dlout(1:2,:,1:end-1)-dlin(12:13,:,1:end-1),dlout(3,:,2:end) - dlout(3,:,1:end-1));
        dldoutdfa   = reshape(dldoutdfa, [numOutputs curr_batch_size*(Nfa-1)]);

        % get random flip angle
        fa_current = randi([1, Nfa], 1, curr_batch_size);

        % extract single point data given the random FA above
        dlResponse = zeros(numOutputs,curr_batch_size, 'like', dlout);
        for k = 1:size(dlout,1)
            ind = sub2ind(size(dlout),k*ones(1,curr_batch_size),1:curr_batch_size,fa_current);
            dlResponse(k,:) = dlout(ind);
%             dlResponse = cat(1,dlResponse,dlout(ind));
        end
        dlFeatures = zeros(numInputs,curr_batch_size, 'like', dlout);
        for k = 1:size(dlin,1)
            ind = sub2ind(size(dlin),k*ones(1,curr_batch_size),1:curr_batch_size,fa_current);
            dlFeatures(k,:) = dlin(ind);
%             dlFeatures = cat(1,dlFeatures,dlin(ind));
        end
        dlFeatures  = dlarray(dlFeatures,'CB');
        dlResponse  = dlarray(dlResponse,'CB');

        % full steady-state curve
        % dlin  - dim1: features; dim2: sample-FA
        % dlout - dim1: pool; dim2: sample-FA
        dlin    = reshape(dlin, [size(dlin,1) size(dlin,2)*size(dlin,3)]);
        dlout   = reshape(dlout,[size(dlout,1) size(dlout,2)*size(dlout,3)]);
        dlin    = dlarray(dlin,'CB');
        dlout   = dlarray(dlout,'CB');

        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function.
        lambda = lambda_curr;
        [gradients,loss] = dlfeval(accfun,parameters,dlFeatures,dlResponse,dlin,dlout,dldoutdfa,lambda,alpha);

        % Update learning rate.
        learningRate = initialLearnRate / (1+decayRate*iteration);

        % Update the network parameters using the adamupdate function.
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);
        
    end

    % Plot training progress.
    loss = double(gather(extractdata(loss)));
    addpoints(lineLoss,iteration, loss);

    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
    drawnow
    
    % print training progress every 10 epoches
    if mod(epoch,10) == 0

        nrse = sqrt(sum((dlResponse - mlp_model_leakyRelu(parameters,dlFeatures,alpha)).^2,'all'))/sqrt(sum((dlResponse ).^2,'all'));
        fprintf('Epoch: %i , NRSE: %f \n',epoch, double(gather(extractdata(nrse))));

        dlnet.parameters    = parameters;
        dlnet.epoch         = epoch;
        save(fullfile(output_dir,[output_prefix '_epoch_' num2str(epoch)]),'dlnet');

        exportgraphics(gcf,fullfile(output_dir,['loss_convergence_epoch_' num2str(epoch) '.png']))
        
    end
end
dlnet.parameters    = parameters;
dlnet.epoch         = epoch;

save(fullfile(output_dir,[output_prefix]),'dlnet','loss');
exportgraphics(gcf,fullfile(output_dir,['loss_convergence.png']))

%% validation part 1
load(fullfile(output_dir,'MCRMWI_MLP_EPGX_RFphase50_T1M234_magn.mat'));

load(fullfile(epgx_dictionary_dir,['EPGX_steadystate_batch-' num2str(44) '.mat']));

Nfa = 40;
fa = linspace(1,90,Nfa);
T1Myelin = 234e-3;

nfeatures = 11;
% features - dim1: nsample; dim2: nFA; dim3: nfeatures 
features_valid    = zeros(size(input_parameter,1),Nfa,nfeatures, 'single');
S_fw_BM     = zeros(size(input_parameter,1),Nfa);
S_m_BM      = zeros(size(input_parameter,1),Nfa);
for kfa = 1:numel(fa)
features_valid(:,kfa,:) = feature_preprocess_MCRMWI_MLP_EPGX_leakyrelu(input_parameter(:,1),input_parameter(:,2),...
                                              input_parameter(:,3),deg2rad(fa(kfa)),input_parameter(:,4),input_parameter(:,5),T1Myelin).';

[S_m, S_fw]    = gpuMCRMWI.model_BM_2T1_analytical(input_parameter(:,4),deg2rad(fa(kfa)),input_parameter(:,1),1./input_parameter(:,2),1/T1Myelin,input_parameter(:,3));

S_fw_BM(:,kfa)  = S_fw;
S_m_BM(:,kfa)   = S_m;
end
response_valid = single(cat(2,  abs(output_parameter(:,:,1)) - S_fw_BM, ...   % IEW
                                abs(output_parameter(:,:,2)) - S_m_BM));   

% features_valid = features_valid(:,:,1:11);
features_valid = reshape(features_valid,[size(features_valid,1)*size(features_valid,2) size(features_valid,3)]).';
% response_valid = reshape(response_valid,[size(response_valid,1)*size(response_valid,2) size(response_valid,3)]).';

features_valid  = gpuArray( dlarray(features_valid,'CB'));
predict_valid   = mlp_model_leakyRelu(dlnet.parameters,features_valid,alpha);

predict_valid = reshape(predict_valid,[size(predict_valid,1) size(input_parameter,1) Nfa]);
predict_valid = reshape( permute(predict_valid,[2 3 1]),[size(input_parameter,1) Nfa*numOutputs]);

nrse = sqrt(sum((response_valid - predict_valid).^2,'all'))/sqrt(sum((response_valid ).^2,'all'));
fprintf('NRSE: %f \n', double(gather(extractdata(nrse))));
