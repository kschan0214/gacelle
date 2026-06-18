
function [S0MW,S0IW,S0EW] = gremwi_S0_compartments(S0,MWF,IWF)

    S0MW = S0 .* MWF;
    S0IW = S0 .* (1-MWF) .* IWF;
    S0EW = S0 .* (1-MWF) .* (1-IWF);
end