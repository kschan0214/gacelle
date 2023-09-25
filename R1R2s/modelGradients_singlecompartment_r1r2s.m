function [gradients,loss] = modelGradients_singlecompartment_r1r2s(parameters,dlR, te, fa, tr, mask)

nTE = size(te,4);
nFA = size(fa,5);

if nargin < 6
    mask = ones(size(dlR,1),size(dlR,2),size(dlR,2),nTE,nFA);
end

% Make predictions with the initial conditions.
R = model_singlecompartment_r1r2s(parameters,te,fa,tr);

R   = dlarray(R(mask>0).',     'CB');
dlR = dlarray(dlR(mask>0).',   'CB');

% compute MSE
loss = mse(R, dlR);

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end