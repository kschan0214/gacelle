function dipoleKernel = dipole_kernel(matrixSize,voxelSize,b0dir)
% Create dipole kernel in k-space with input matrix dimensions
% and spatial resolution
%
% Output
% ______
%   dipoleKernel      : Dipole kernel (in k-space), dimensionless,
%                       real-valued and even, so the adjoint equals
%                       the forward (conj(D) == D)
%
% Kwok-shing Chan @ DCCN
% Date created: 24 March 2017
% Date last modified: 27 September 2017

    if nargin<3
        b0dir = [0 0 1];
    end

    [ky,kx,kz] = meshgrid(-matrixSize(2)/2:matrixSize(2)/2-1, ...
                          -matrixSize(1)/2:matrixSize(1)/2-1, ...
                          -matrixSize(3)/2:matrixSize(3)/2-1);

    kx = (kx / max(abs(kx(:)))) / voxelSize(1);
    ky = (ky / max(abs(ky(:)))) / voxelSize(2);
    kz = (kz / max(abs(kz(:)))) / voxelSize(3);

    k2 = kx.^2 + ky.^2 + kz.^2;

    dipoleKernel = fftshift( 1/3 - (kx*b0dir(1) + ky*b0dir(2) + kz*b0dir(3)).^2 ./ (k2 + eps) );

end