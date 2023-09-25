function [gradients,loss] = modelGradients_singlecompartment_r1r2s_1d(parameters,dlR, te, fa, tr)

% nTE = size(te,2);
% nFA = size(fa,3);

% Make predictions with the initial conditions.
% parameters.r2s = parameters.r2s * 10;   % upscale R2* to the usualy range
R = model_singlecompartment_r1r2s(parameters,te,fa,tr);
% parameters.r2s = parameters.r2s / 10;   % downscale R2* to the optimisation range

R   = dlarray(R(:).',     'CB');
dlR = dlarray(dlR(:).',   'CB');

% compute MSE
loss = mse(R, dlR);

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end