
function s = Example_newmodel_FWD_askadam( pars, mask, t)

    t = t(:);

    % fields = fieldnames(pars);

    % make sure the first dimension is 1
    if ~isempty(mask)
        S0       =  squeeze(pars.S0(mask));
        R2       =  squeeze(pars.R2(mask));
    else
        S0      = pars.S0;
        R2      = pars.R2;
    end
 
    s = S0 .* exp(-t.*R2);

end