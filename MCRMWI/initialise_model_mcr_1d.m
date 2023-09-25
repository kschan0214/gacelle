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
function parameters = initialise_model_mcr_1d(img,totalfield,pini)

% [sx, sy, sz] = size(img);

% img = reshape(img, [sx*sy*sz, nTE, nFA]);

% Initialize the parameters for the first fully connect operation. The first fully connect operation has two input channels.
parameters = struct;

% assume comparment a has long T1 and T2*
parameters.s0iw         = gpuArray( dlarray(max(max(abs(img),[],2),[],3)*0.6));
parameters.s0ew         = gpuArray( dlarray(max(max(abs(img),[],2),[],3)*0.4));
% parameters.s0mw         = gpuArray( dlarray(max(max(abs(img),[],2),[],3)*0.1));
parameters.s0mw         = gpuArray( dlarray(rand(size(img,1), 1,'single') ));

parameters.r1iew        = gpuArray( dlarray(rand(size(img,1), 1,'single') )) ;
parameters.kiewm        = gpuArray( dlarray(zeros(size(img,1),1,'single') )) ;

parameters.totalField   = gpuArray( dlarray(totalfield) );
parameters.pini         = gpuArray( dlarray(pini) );

parameters.r2siw        = gpuArray( dlarray(ones(size(img,1), 1,'single') )) ;
parameters.r2sew        = gpuArray( dlarray(ones(size(img,1), 1,'single') )) ;
parameters.r2smw        = gpuArray( dlarray(ones(size(img,1), 1,'single') )) ;

parameters.freq_mw      = gpuArray( dlarray(ones(size(img,1), 1,'single') )) ;
parameters.freq_iw      = gpuArray( dlarray(zeros(size(img,1), 1,'single') )) ;

end