%% Signal related
function signal = model_jointR1R2s_singlecompartment(m0,r2s,R1,te,TR,fa)
% m0    : proton density weighted signal
% r2s   : R2*, in s^-1 or ms^-1
% R1    : R1, in s^-1 or ms^-1
% te    : echo time, in s or ms
% TR    : repetition time, in s or ms
% fa    : true flip angles, in radian

signal = m0 .* sin(fa) .* (1 - exp(-TR.*R1)) ./ (1 - cos(fa) .* exp(-TR.*R1)) .* ...
                exp(-te .* r2s);


end