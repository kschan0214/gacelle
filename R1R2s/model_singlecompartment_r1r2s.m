function dlU = model_singlecompartment_r1r2s(parameters,te,fa,tr)

% matrix size info
% dims(1) = size(parameters.m0,1);
% dims(2) = size(parameters.m0,2);
% dims(3) = size(parameters.m0,3);
% nTE = size(te,4);
% nFA = size(fa,5);

% if nargin < 8
%     weights = ones([dims nTE nFA]);
% end

dlU = parameters.m0 .* sin(fa) .* (1 - exp(-tr.*parameters.r1)) ./ (1 - cos(fa) .* exp(-tr.*parameters.r1)) .* ...
            exp(-te .* parameters.r2s*10);

end