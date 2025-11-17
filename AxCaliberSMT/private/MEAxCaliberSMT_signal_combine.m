
function s = MEAxCaliberSMT_signal_combine(f,fcsf,Sa,Se,Scsf)

% Combined signal
s = (1-fcsf).*(f.*Sa + (1-f).*Se) + fcsf.*Scsf;

end