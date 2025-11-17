%% S = C_arrayfun(g, C, D, td)
% Description: Auxillary function for arrayfun
%
% Kwok-shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 13 November 2025
% Date modified:
%
%
function S = C_arrayfun(g, C, D, td)

C = C.*D.*g.^2.*td.^3;

S = exp(-C);

end