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
function feature = feature_preprocess_MCRMWI_MLP_EPGX_leakyrelu(fm,t1iew,kiewmw,fa,tr,t2iew,t1mw)

numSample = max([ numel(fm) numel(t1iew) numel(t1iew) numel(kiewmw) numel(fa) numel(tr) numel(t2iew)]);

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

feature = zeros(numel(fm),11,'like',fm);

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
feature(:,9)    = model_GRE_steadystate_Bloch(1-fm(:),t1iew(:),tr(:),fa(:));       % steady-state of IEW
feature(:,10)   = model_GRE_steadystate_Bloch(fm(:),t1mw,tr(:),fa(:));             % steady-state of M

% new
feature(:,11)   = exp(-tr(:) .* (1./t2iew));                % exp(-TR/T2iew)

% [S_fw, S_m]     = model_GRE_steadystate_BM(fm(:),1./t1iew(:),1/t1mw,kiewmw(:),fa(:),tr(:));
% feature(:,12)   = S_fw;             % BM steady-state of IEW
% feature(:,13)   = S_m;              % BM steady-state of M
% 
% % in case kmiew is very fast due to low concentration then = single compartment
% feature(feature(:,6) < eps,12) = feature(feature(:,6) < eps,9);
% feature(feature(:,6) < eps,13) = feature(feature(:,6) < eps,10);

% feature(:,9)    =  signal_steadystate_Bloch(1-fm(:),1./t1iew(:),fa(:),tr(:));   
% feature(:,10)   =  signal_steadystate_Bloch(fm(:),1./t1mw,fa(:),tr(:));         % steady-state of MW

% feature(:,3) = exp(-tr(:)./t2iw(:));                        % exp(-TR/t2f)
% feature(:,4) = exp(-tr(:)./t2mw(:));                        % exp(-TR/t2s)
% feature(:,6) = 2*pi*dfreq(:)/(B0*42.58);                    % rad ppm
% feature(:,10) = exp(-tr(:)./t2iw(:)) - exp(-tr(:)./t2mw(:)); % exp(-TR/t2f) - exp(-TR/t2s)
% feature(:,11) = exp(-tr(:)./t1iew(:)).* sin(fa(:));        % exp(-TR/T1)*sin(alpha)

feature = feature.';

end

function s = model_GRE_steadystate_Bloch(m0,t1,tr,alpha)
    E1 = exp(-tr./t1);
    s = (m0.*(1-E1).*sin(alpha))./(1-E1.*cos(alpha));
end

function [S_fw, S_m] = model_GRE_steadystate_BM(f,R1f,R1r,kfr,fa,TR)
        % f : myelin volume fraction
        % R1f : R1 of free (IE) water
        % kfr : exchange rate from free (IE) water to myelin
        % true_famp: true flip angle map in radian 


        % BM free water steady-state signal
        S_fw = ((f./2 - 1./2).*(kfr.*sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f - ...
            (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + ...
            R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) - ...
            2.*sin(2.*fa).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
            2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) + ...
            2.*kfr.*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
            2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa) + ...
            sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
            2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*(R1f.^2.*f.^2 - ...
            2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) + ...
            2.*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
            4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
            4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) - ...
            2.*kfr.*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + ...
            R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa) - ...
            kfr.*sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
            4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) - ...
            4.*exp((TR.*(kfr + R1f.*f + R1r.*f))./f).*sin(fa).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
            2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) + 2.*exp((TR.*(kfr + R1f.*f + ...
            R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + ...
            2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
            2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) + ...
            sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
            2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*(R1f.^2.*f.^2 - ...
            2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) - ...
            R1f.*f.*sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + ...
            R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) + R1r.*f.*sin(2.*fa).*exp((TR.*(kfr + R1f.*f + ...
            R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + ...
            kfr.^2).^(1./2)))./(2.*f)) - 2.*R1f.*f.*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
            2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa) + ...
            2.*R1r.*f.*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
            2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa) + ...
            2.*R1f.*f.*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + ...
            R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa) + ...
            R1f.*f.*sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
            2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) - ...
            2.*R1r.*f.*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + ...
            R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa) - ...
            R1r.*f.*sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
            2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f))))./(2.*(exp((TR.*(kfr + R1f.*f + ...
            R1r.*f))./f) + cos(fa).^2 - exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
            2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*cos(fa) - exp((TR.*(kfr + ...
            R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + ...
            2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*cos(fa)).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2));
        
        % BM myelin semi-solid steady-state signal
        S_m = -(2.*f.*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + ...
            R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
            4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) - 2.*f.*sin(2.*fa).*(R1f.^2.*f.^2 - ...
            2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) - ...
            2.*f.*kfr.*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
            4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa) - f.*kfr.*sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f + ...
            (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) - ...
            4.*f.*exp((TR.*(kfr + R1f.*f + R1r.*f))./f).*sin(fa).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
            4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) - 2.*R1f.*f.^2.*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
            4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa) - ...
            R1f.*f.^2.*sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
            4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) + 2.*R1r.*f.^2.*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
            4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa) + ...
            R1r.*f.^2.*sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
            4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) + 2.*f.*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
            4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa).*(R1f.^2.*f.^2 - ...
            2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) + ...
            f.*sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
            4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
            4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) + f.*kfr.*sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
            4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) + 2.*f.*kfr.*exp((TR.*(kfr + R1f.*f + ...
            R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + ...
            2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa) + R1f.*f.^2.*sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - ...
            2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) - ...
            R1r.*f.^2.*sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
            4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)) + f.*sin(2.*fa).*exp((TR.*(kfr + R1f.*f + R1r.*f - ...
            (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + ...
            2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
            4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2) + 2.*R1f.*f.^2.*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
            4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa) - ...
            2.*R1r.*f.^2.*exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - ...
            4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*sin(fa))./(4.*(exp((TR.*(kfr + R1f.*f + R1r.*f))./f) + cos(fa).^2 - ...
            exp((TR.*(kfr + R1f.*f + R1r.*f - (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + ...
            2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*cos(fa) - exp((TR.*(kfr + R1f.*f + R1r.*f + (R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + 4.*R1f.*f.^2.*kfr - ...
            2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2)))./(2.*f)).*cos(fa)).*(R1f.^2.*f.^2 - 2.*R1f.*R1r.*f.^2 + ...
            4.*R1f.*f.^2.*kfr - 2.*R1f.*f.*kfr + R1r.^2.*f.^2 - 4.*R1r.*f.^2.*kfr + 2.*R1r.*f.*kfr + kfr.^2).^(1./2));
         
    
    end