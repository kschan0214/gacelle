function [gradients,loss] = modelGradients_mcr_1d(parameters,dlR, te, fa, tr, weights, r1mw,rho_mw, dlnet,parameter_scale,mask)

% nTE = size(te,2);
% nFA = size(fa,3);

% Make predictions with the initial conditions.
% R = model_mcr_ann_real_imag_normalised(parameters,te,fa,tr,dlnet);
[dlR_real, dlR_imag] = model_mcr_ann_1d(parameters,te,fa,tr,r1mw,rho_mw,dlnet,weights,parameter_scale);
% R = model_mcr_ann_1d(parameters,te,fa,tr,r1mw,rho_mw,dlnet,weights,parameter_scale);

% R   = dlarray(dlR_real(:).',    'CB');
% dlR = dlarray(real(dlR(:)).',   'CB');

% compute MSE
% loss_real = mse(dlarray((dlR_real(:).*weights(:)).', 'CB'), dlarray(real(dlR(:)).', 'CB'));
% loss_imag = mse(dlarray((dlR_imag(:).*weights(:)).', 'CB'), dlarray(imag(dlR(:)).', 'CB'));
% loss = loss_real + loss_imag;
% loss = mse( dlarray( cat(1,dlR_real(:).*weights(:),dlR_imag(:).*weights(:)).',    'CB'),...
%             dlarray( cat(1,real(dlR(:)).*weights(:),imag(dlR(:)).*weights(:)).',  'CB'));
% R   = dlarray(R, 'CB');
% dlR = dlarray(dlR, 'CB');
R = dlarray(cat(1,dlR_real(mask>0),dlR_imag(mask>0)).', 'CB');
dlR = dlarray(cat(1,real(dlR(mask>0)),imag(dlR(mask>0))).', 'CB');

% compute MSE
loss = mse(R, dlR);

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end