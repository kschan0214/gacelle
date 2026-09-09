%% parameters = create_mlp(numLayers,numNeurons,numInputs,numOutputs)
%
% Input
% --------------
% numLayers     : no. of fully connected layers
% numNeurons    : no. of neurons per layer, can be either scalar or vector
% numInputs     : no. of input
% numInputs     : no. of output
%
% Output
% --------------
% parameters    : structure contains all trainable network parameters
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
function parameters = create_mlp(numLayers,numNeurons, numInputs,numOutputs)

if isempty(numLayers)
    numLayers = length(numNeurons) + 1;
end

if isscalar(numNeurons)
    numNeurons = ones(1,numLayers)*numNeurons;
end

% Initialize the parameters for the first fully connect operation. The first fully connect operation has two input channels.
parameters = struct;

sz                      = [numNeurons(1) numInputs];
parameters.fc1.Weights  = initializeHe(sz,numInputs);
parameters.fc1.Bias     = initializeZeros([numNeurons(1) 1]);

for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz                          = [numNeurons(layerNumber) numNeurons(layerNumber-1)];
    numIn                       = numNeurons(layerNumber-1);
    parameters.(name).Weights   = initializeHe(sz,numIn);
    parameters.(name).Bias      = initializeZeros([numNeurons(layerNumber) 1]);
end

% Initialize the parameters for the final fully connect operation. The final fully connect operation has one output channel.
sz                                      = [numOutputs numNeurons(numLayers-1)];
numIn                                   = numNeurons(numLayers-1);
parameters.("fc" + numLayers).Weights   = initializeHe(sz,numIn);
parameters.("fc" + numLayers).Bias      = initializeZeros([1 numOutputs]);

end

function parameter = initializeHe(sz,numIn,className)

arguments
    sz
    numIn
    className = 'single'
end

parameter = sqrt(2/numIn) * randn(sz,className);
parameter = dlarray(parameter);

end

function parameter = initializeZeros(sz,className)

arguments
    sz
    className = 'single'
end

parameter = zeros(sz,className);
parameter = dlarray(parameter);

end