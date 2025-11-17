% function q = gacelle_trapz(y, x, dim)
% substitution function of matlab trapz that is compatible with GACELLE
%   q = TRAPZ_ND(y)                     % unit spacing (dx = 1), last non-singleton dim
%   q = TRAPZ_ND(y, x)                  % x = scalar dx OR vector of coordinates
%   q = TRAPZ_ND(y, x, dim)             % specify dim explicitly
%
%   - If x is a scalar, it is interpreted as uniform spacing dx.
%   - If x is a vector, its length must match size(y, dim).
function q = gacelle_trapz(y, x, dim)

% check dimension input
if nargin < 3 || isempty(dim)
    sz  = size(y);
    dim = find(sz > 1, 1, 'last');  % last non-singleton dimension
    if isempty(dim), dim = 1; end
end

% check x input
if nargin < 2 || isempty(x)
    uniform = true; dx = 1;
elseif isscalar(x)
    uniform = true; dx = x;
else
    uniform = false;
    if numel(x) ~= size(y, dim)
        error('Length of x (%d) must match size(y, dim) = %d.', numel(x), size(y, dim));
    end
end

n = size(y, dim);
if n < 2
    % Nothing to integrate; return zeros with size 1 along 'dim'
    q = sum(zeros(size(y), 'like', y), dim);
    return
end

    % Build indexers for slices [1..n-1] and [2..n] along 'dim'
    idxA        = repmat({':'}, 1, ndims(y)); idxB = idxA;
    idxA{dim}   = 1:(n-1);
    idxB{dim}   = 2:n;

    if uniform
        q = sum( (y(idxA{:}) + y(idxB{:})) .* (dx/2), dim );
    else
        % Nonuniform spacing: use segment widths diff(x)
        w   = diff(x);                       % [n-1]
        shp = ones(1, ndims(y)); shp(dim) = numel(w);
        w   = reshape(w, shp);               % broadcast along other dims
        q   = sum( (y(idxA{:}) + y(idxB{:})) .* (w/2), dim );
    end

    % dx = x(2) - x(1);
    % 
    % if ndims(y) == 3
    %     q = sum((y(:,:,2:end) + y(:,:,1:end-1)) * (dx) / 2, ndims(x));
    % elseif ndims(y) == 4
    %     q = sum((y(:,:,:,2:end) + y(:,:,:,1:end-1)) * (dx) / 2, ndims(x));
    % end

end