function [dlU_real, dlU_imag] = model_mcr_ann_1d(parameters,te,fa,tr,r1mw,rho_mw,dlnet,weights,parameter_scale)
% function dlU = model_mcr_ann_1d(parameters,te,fa,tr,r1mw,rho_mw,dlnet,weights,parameter_scale)

numSample = size(parameters.s0iw,1);
nTE = size(te,2);
nFA = size(fa,3);

% if nargin < 8
%     weights = ones([dims nTE nFA]);
% end

% scaling
parameters.r2siw    = parameters.r2siw * parameter_scale.r2siw;
parameters.r2sew    = parameters.r2sew * parameter_scale.r2sew;
parameters.r2smw    = parameters.r2smw * parameter_scale.r2smw;
parameters.freq_mw  = parameters.freq_mw * parameter_scale.freq_mw;


scaleFactor = parameters.s0iw + parameters.s0ew + parameters.s0mw / rho_mw;
fx = (parameters.s0mw / rho_mw) ./ scaleFactor;

ss = zeros(2,numSample,nFA, 'like', parameters.r2siw );
for k = 1:nFA
features = feature_preprocess_LeeANN4MWImodel(fx(:),1./parameters.r1iew(:),...
                                              parameters.kiewm(:),fa(:,1,k),tr,1./r1mw);
features    = gpuArray( dlarray(features,'CB'));

ss(:,:,k) = reshape(mlp_model_leakyRelu(dlnet.parameters,features,dlnet.alpha),[2 numSample]);
end
ss = permute( ss, [2 1 3]);
% features = feature_preprocess_LeeANN4MWImodel(repmat(fx(:),[nFA 1]),repmat(1./parameters.r1iew(:),[nFA 1]),...
%                                               repmat(parameters.kiewm(:),[nFA 1]),fa(:),tr,1./r1mw);
% features    = gpuArray( dlarray(features,'CB'));
% 
% ss = mlp_model_leakyRelu(dlnet.parameters,features,dlnet.alpha);
% ss = permute( reshape(ss,[2 numSample nFA]), [2 1 3]);
% ss_mask     = mlp_model_leakyRelu(dlnet.parameters,features,dlnet.alpha);
% ss              = zeros([2 numSample*nFA], 'like', parameters.s0iw);
% ss(:,mask(:,:,:,1,:)>0)    = ss_mask;
% ss              = scaleFactor.* permute( reshape(ss,[2 dims nFA]),[2 3 4 1 5]);

dlU_real =  (ss(:,1,:).*(parameters.s0ew./(parameters.s0iw+parameters.s0ew)) .* exp(-te .*  parameters.r2sew )) .* ...
                cos(2.*pi.*parameters.totalField.*te + parameters.pini) + ... % EW 
                ss(:,1,:).*(parameters.s0iw./(parameters.s0iw+parameters.s0ew)) .* exp(-te .* (parameters.r2siw)) .* ...
                cos(2*pi*parameters.freq_iw.*te + 2.*pi.*parameters.totalField.*te + parameters.pini) + ...
                ss(:,2,:).*rho_mw .* exp(-te .* (parameters.r2smw)) .* ...
                cos(2*pi*parameters.freq_mw.*te + 2.*pi.*parameters.totalField.*te + parameters.pini) ; 

dlU_imag =  (ss(:,1,:).*(parameters.s0ew./(parameters.s0iw+parameters.s0ew)) .* exp(-te .*  parameters.r2sew )) .* ...
                sin(2.*pi.*parameters.totalField.*te + parameters.pini) + ... % EW 
                ss(:,1,:).*(parameters.s0iw./(parameters.s0iw+parameters.s0ew)) .* exp(-te .* (parameters.r2siw)) .* ...
                sin(2*pi*parameters.freq_iw.*te + 2.*pi.*parameters.totalField.*te + parameters.pini) + ...
                ss(:,2,:).*rho_mw .* exp(-te .* (parameters.r2smw)) .* ...
                sin(2*pi*parameters.freq_mw.*te + 2.*pi.*parameters.totalField.*te + parameters.pini) ; 

% dlU_real =  (ss(:,:,:,1,:).*(parameters.s0ew./(parameters.s0iw+parameters.s0ew)) .* exp(-te .*  parameters.r2sew )) .* ...
%         cos(2.*pi.*permute(parameters.totalField,[1 2 3 5 4]).*te + parameters.pini) + ... % EW 
%         ss(:,:,:,1,:).*(parameters.s0iw./(parameters.s0iw+parameters.s0ew)) .* exp(-te .* (parameters.r2siw)) .* ...
%         cos(2*pi*parameters.freq_iw.*te + 2.*pi.*permute(parameters.totalField,[1 2 3 5 4]).*te + parameters.pini) + ...
%         ss(:,:,:,2,:).*rho_mw .* exp(-te .* (parameters.r2smw)) .* ...
%         cos(2*pi*parameters.freq_mw.*te + 2.*pi.*permute(parameters.totalField,[1 2 3 5 4]).*te + parameters.pini) ; 

% dlU_imag =  (ss(:,:,:,1,:).*(parameters.s0ew./(parameters.s0iw+parameters.s0ew)) .* exp(-te .*  parameters.r2sew )) .* ...
%             sin(2.*pi.*permute(parameters.totalField,[1 2 3 5 4]).*te + parameters.pini) + ... % EW 
%             ss(:,:,:,1,:).*(parameters.s0iw./(parameters.s0iw+parameters.s0ew)) .* exp(-te .* (parameters.r2siw)) .* ...
%             sin(2*pi*parameters.freq_iw.*te + 2.*pi.*permute(parameters.totalField,[1 2 3 5 4]).*te + parameters.pini) + ...
%             ss(:,:,:,2,:).*rho_mw .* exp(-te .* (parameters.r2smw)) .* ...
%             sin(2*pi*parameters.freq_mw.*te + 2.*pi.*permute(parameters.totalField,[1 2 3 5 4]).*te + parameters.pini) ; 

dlU_real(isnan(dlU_real)) = 0;
dlU_imag(isnan(dlU_imag)) = 0;
dlU_real = dlU_real.*weights;
dlU_imag = dlU_imag.*weights;
% dlU = cat(1,dlU_real(:),dlU_imag(:)).';

end