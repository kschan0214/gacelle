function fvf = hcfm_fibre_volume_fraction(Va,Ve,Vm)
    Va  = Va ./ (Va+Ve+Vm);
    Vm  = Vm ./ (Va+Ve+Vm);
    fvf = Va+Vm; 
end