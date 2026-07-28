%% Signal related
function signal = model_R2s_singlecompartment(m0,r2s,te)
% m0    : proton density weighted signal
% r2s   : R2*, in s^-1 or ms^-1
% te    : echo time, in s or ms

signal = m0 .* exp(-te .* r2s);


end