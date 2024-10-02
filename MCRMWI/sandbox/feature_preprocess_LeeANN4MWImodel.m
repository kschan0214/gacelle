%% feature = feature_preprocess_LeeANN4MWImodel(fm,t1iew,kiewmw,fa,tr,t1mw)
%
% Input
% --------------
% fm            : myelin fraction, in ratio
% t1iew         : IEW T1, in second
% kiewmw        : exchange rate from IEW to M, s^-1
% fa            : flip angle, in radian
% tr            : TR, in s
% t1mw          : myelin T1, in second
%
% Output
% --------------
% feature       : structure contains all trainable network parameters
%
% Description: convert the input parameter to the features of ANN
%
% Kwok-shing Chan @ DCCN
% kwokshing.chan@donders.ru.nl
% Date created: 28 October 2021
% Date modified:
%
%
function feature = feature_preprocess_LeeANN4MWImodel(fm,t1iew,kiewmw,fa,tr,t1mw)

numSample = max([ numel(fm) numel(t1iew) numel(t1iew) numel(kiewmw) numel(fa) numel(tr)]);

if nargin < 6
    t1mw = 234e-3;
end

if isscalar(fm)
    fm = ones(numSample,1) * fm;
end

if isscalar(t1iew)
    t1iew = ones(numSample,1) * t1iew;
end

if isscalar(kiewmw)
    kiewmw = ones(numSample,1) * kiewmw;
end

if isscalar(tr)
    tr = ones(numSample,1) * tr;
end

if isscalar(fa)
    fa = ones(numSample,1) * fa;
end

kmwiew = kiewmw(:) .* (1-fm(:))./fm(:);
kmwiew(fm(:)==0) = 0;

feature = zeros(numel(fm),10,'like',fm);

% direct conversion
feature(:,1) = fm(:);                                       % MWF (or myelin fraction)
feature(:,2) = exp(-tr(:)./t1iew(:));                       % exp(-TR/T1), i.e. E1
feature(:,3) = exp(-tr(:).*kiewmw(:));                      % exp(-TR*kab)
feature(:,4) = sin(fa(:));                                  % sin(alpha)

% further derived
feature(:,5) = 1 - feature(:,2) .* cos(fa(:));              % steady-state denominator: 1-E1*cos(alpha); 
feature(:,6) = exp(-tr(:) .* kmwiew(:));                    % exp(-TR*kba)
feature(:,7) = exp(-tr(:) .* (1./t1iew(:) + kiewmw(:)));    % exp(-TR* (R1a+kab))
feature(:,8) = exp(-tr(:) .* (1./t1mw + kmwiew(:)));        % exp(-TR* (R1b+kba))

% steady-state Bloch closed form solution
feature(:,9)    =  signal_steadystate_Bloch(1-fm(:),1./t1iew(:),fa(:),tr(:));   % steady-state of IEW
feature(:,10)   =  signal_steadystate_Bloch(fm(:),1./t1mw,fa(:),tr(:));         % steady-state of MW

% feature(:,3) = exp(-tr(:)./t2iw(:));                        % exp(-TR/t2f)
% feature(:,4) = exp(-tr(:)./t2mw(:));                        % exp(-TR/t2s)
% feature(:,6) = 2*pi*dfreq(:)/(B0*42.58);                    % rad ppm
% feature(:,10) = exp(-tr(:)./t2iw(:)) - exp(-tr(:)./t2mw(:)); % exp(-TR/t2f) - exp(-TR/t2s)
% feature(:,11) = exp(-tr(:)./t1iew(:)).* sin(fa(:));        % exp(-TR/T1)*sin(alpha)

feature = feature.';

end