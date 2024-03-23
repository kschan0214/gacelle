%% function parameters = initialise_model_singlecompartment_r1r2s_1d(img)
%
% Input
% --------------
%
% Output
% --------------
%
% Description:
%
% Kwok-shing Chan @ DCCN
% k.chan@donders.ru.nl
% Date created: 24 October 2021
% Date modified:
%
%
function parameters = initialise_model_singlecompartment_r1r2s_1d(numvoxel)

% Initialize the parameters for the first fully connect operation. The first fully connect operation has two input channels.
parameters = struct;

% assume comparment a has long T1 and T2*
% it is important to have all parameters in the same range to get similar
% stepp size
parameters.m0   = gpuArray( dlarray(rand(numvoxel,1,'single') )) ;
parameters.r1   = gpuArray( dlarray(rand(numvoxel,1,'single') )) ;
parameters.r2s  = gpuArray( dlarray(rand(numvoxel,1,'single') )) ;

end