%% ss = signal_steadystate_Bloch(m0,r1,fa,tr)
%
% Input
% --------------
% m0            : signal intensity
% r1            : R1, in s^-1
% fa            : flip angle, in radian
% tr            : TR, in s
%
% Output
% --------------
% ss            : steady-state signal
%
% Description: create steady-state signal based on closed form solution of
% the Bloch equation. Compatible for matrix based operation
%
% Kwok-shing Chan @ DCCN
% kwokshing.chan@donders.ru.nl
% Date created: 28 October 2021
% Date modified:
%
%
function ss = signal_steadystate_Bloch(m0,r1,fa,tr)

ss = m0 .* sin(fa) .* (1 - exp(-tr.*r1)) ./ (1 - cos(fa) .* exp(-tr.*r1));

end