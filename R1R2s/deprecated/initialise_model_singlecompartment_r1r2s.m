%% function parameters = initialise_model_r1r2s(img)
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
function parameters = initialise_model_singlecompartment_r1r2s(img)

[sx, sy, sz] = size(img);

% Initialize the parameters for the first fully connect operation. The first fully connect operation has two input channels.
parameters = struct;

% assume comparment a has long T1 and T2*
parameters.m0   = gpuArray( dlarray(rand(sx,sy,sz,'single') )) ;
parameters.r1   = gpuArray( dlarray(rand(sx,sy,sz,'single') )) ;
parameters.r2s  = gpuArray( dlarray(rand(sx,sy,sz,'single') )) ;

end