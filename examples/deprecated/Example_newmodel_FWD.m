
function s = Example_newmodel_FWD( pars, t)

    t = t(:);

    fields = fieldnames(pars);

    % make sure the first dimension is 1
    if size(pars.(fields{1}),1) ~= 1
        S0       = shiftdim(pars.S0,-1);
        R2       = shiftdim(pars.R2,-1);
    else
        S0      = pars.S0;
        R2      = pars.R2;
    end
 
    s = S0 .* exp(-t.*R2);

end