%% [gradients,loss] = modelGradients_mlp_epgx_LeeANN4MWImodel(parameters,dlX,dlR,dlXf,dlRf,dldRf,istanh,lambda,alpha)
%
% Input
% --------------
% parameters    : structure contains all trainable network parameters
% dlX           : input features
% dlR           : output response
% dlXf          : input feature, FA from 1-90 degrees
% dlRf          : output response, FA from 1-90 degrees
% dldRfdfa      : 1st derivative of dlXf w.r.t FA
% lambda        : regularisation parameter for MSE loss of dldRfdfa
% alpha         : leaky relu scale factor
%
% Output
% --------------
% gradients     : gradient
% loss          : MSE loss
%
% Description: compute gradient and loss for training
%
% Kwok-shing Chan @ DCCN
% kwokshing.chan@donders.ru.nl
% Date created: 28 October 2021
% Date modified:
%
%
function [gradients,loss] = modelGradients_mlp_epgx_ANN_corr_L1(parameters,dlX,dlR,dlXf,dlRf,dldRfdfa,lambda,alpha)

if nargin < 8
    alpha = 0.01;
end

nsample = size(dlX,2);
nFA     = size(dlXf,2)/nsample;
nOutput = size(dlR,1);

% Make predictions given single point.
U = mlp_model_leakyRelu(parameters,dlX,alpha);

% Calculate loss_p enforcing the point-wise accuracy
loss_p = l1loss(U, dlR);

% make predictions for full steady-state curve
U_full = mlp_model_leakyRelu(parameters,dlXf,alpha);

% Calculate loss_f enforcing the full spectrum accuracy
loss_f = l1loss(U_full, dlRf);

% dS/dFA, i.e. gradient w.r.t flip angle
U_full      = reshape(U_full, [nOutput, nsample, nFA]);
dU_fulldfa  = U_full(:,:,2:end) - U_full(:,:,1:end-1);
dU_fulldfa  = reshape(dU_fulldfa, [nOutput nsample*(nFA-1)]);
dU_fulldfa  = dlarray(dU_fulldfa, 'CB');

% Calculate loss_g enforcing the the smoothness of the full spectrum based
% on 1st derivative
loss_g = l1loss(dU_fulldfa,dldRfdfa);

loss = loss_p + loss_f + lambda*loss_g;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end