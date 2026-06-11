function s = signal_SANDI(fs,f,Ssoma,Sneurite,Sextra)
    s = fs.*Ssoma + (1-fs).*f.*Sneurite + (1-fs).*(1-f).*Sextra;
end