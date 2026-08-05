function Gx = fgrad(chi, voxelSize)
%FGRAD  Discrete gradient using FORWARD differences, Neumann boundary.
%
%   Gx = this.fgrad(chi)              gradient of the 3D array chi
%   Gx = this.fgrad(chi, voxelSize)   scaled by 1/voxelSize(k)
%
% Input
% -----
%   chi        : 3D array. Also accepts gpuArray and dlarray.
%   voxelSize  : [dx dy dz], default [1 1 1]
%
% Output
% ------
%   Gx         : [size(chi) 3], components concatenated along dim 4
%
% D(i) = (chi(i+1) - chi(i))/h for i < n, and D(n) = 0. The trailing
% zero is the Neumann (zero-flux) boundary condition, obtained in the
% original by padding with the last slice before subtracting.
%
% PREFER THIS OVER cgrad FOR TV REGULARISATION.
% Forward differences have a trivial null space (constants only),
% whereas the central difference used by cgrad also annihilates the
% alternating mode (+1,-1,+1,...), so a central-difference TV penalty
% does not suppress checkerboard noise and decouples the odd and even
% sublattices. cgrad remains the right choice for deriving an edge
% mask from a magnitude image, where the half-voxel shift of a
% forward difference would misregister edges against the grid.
%
% The adjoint of this operator is the negative backward divergence
% (bdiv in the MEDI toolbox). It is not needed here - dlgradient
% supplies it - but an exactly matched adjoint is required if this
% operator is ever reused inside a CG-type solver.
%
% Original: Youngwook Kee (Oct 2015), MEDI toolbox.
% References
%   [1] Chambolle. An Algorithm for Total Variation Minimization and
%       Applications. JMIV 2004.
%   [2] Pock et al. Global Solutions of Variational Models with
%       Convex Regularization. SIIMS 2010.

    if nargin < 3 || isempty(voxelSize)
        voxelSize = [1 1 1];
    end
    if numel(voxelSize) < 3
        voxelSize = [voxelSize(:).' ones(1, 3-numel(voxelSize))];
    end

    Gx = cat(4, diff_forward(chi, 1, voxelSize(1)), ...
                diff_forward(chi, 2, voxelSize(2)), ...
                diff_forward(chi, 3, voxelSize(3)));

end

function D = diff_forward( x, dim, h)
% Forward difference along `dim`, zero at the trailing slice.
%
% Built by slicing and concatenation rather than by padding then
% subtracting: the original materialised a full shifted copy of chi
% per dimension before the subtraction, this allocates only the
% result. Fixed-index slicing traces cleanly under dlarray and
% dlaccelerate, and mirrors diff_central so the two stay comparable.

    n = size(x, dim);

    if n < 2
        D = x * 0;                  % *0 rather than zeros(...,'like',x)
        return                      % to preserve dlarray/gpuArray type
    end

    idx = repmat({':'}, 1, max(ndims(x), 3));

    interior = slice(x, idx, dim, 2:n) - slice(x, idx, dim, 1:n-1);
    trailing = slice(x, idx, dim, n) * 0;      % Neumann BC

    D = cat(dim, interior, trailing) / h;

end

function y = slice(x, idx, dim, k)
    idx{dim} = k;
    y = x(idx{:});
end