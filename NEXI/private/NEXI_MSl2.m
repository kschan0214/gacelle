%% M = NEXI_MSl2(M,x)
%
% Description: Support function for gpuNEXImcmc to be able to use arrayfunc
%              Compute the combined dMRI signal ffrom all compartments
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 17 June 2024
% Date modified: 
%
%
function M = NEXI_MSl2(M,x)

M = M.*(3*x.^2-1)/2; 

end