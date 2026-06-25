%% dlU = mlp_model_leakyRelu(parameters,dlX,scaleFactor)
%
% Input
% --------------
% parameters    : structure contains all trainable network parameters
% dlX           : input data of the ANN
% scaleFactor   : no. of input
% numInputs     : no. of output
%
% Output
% --------------
% dlU           : network predicted output
%
% Description: create trainable multi-layer perceptron netowrk parameters for customised training
%              loop
%
% Kwok-shing Chan @ DCCN
% kwokshing.chan@donders.ru.nl
% Date created: 28 October 2021
% Date modified:
%
%
function dlU = mlp_model_leakyRelu(parameters,dlX,scaleFactor)

if nargin < 3
    scaleFactor = 0.01;
end

numLayers = numel(fieldnames(parameters));

% First fully connect operation.
weights = parameters.fc1.Weights;
bias    = parameters.fc1.Bias;
dlU     = fullyconnect(dlX,weights,bias);

% leaky RELU and fully connect operations for remaining layers.
for i=2:numLayers
    name = "fc" + i;

    dlU = leakyrelu(dlU, scaleFactor);
    
    weights = parameters.(name).Weights;
    bias    = parameters.(name).Bias;
    dlU     = fullyconnect(dlU, weights, bias);
end

end