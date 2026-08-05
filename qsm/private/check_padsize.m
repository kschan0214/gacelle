function [padPre, padPost, matrixSizePad] = check_padsize( mask, gapMinMM, voxelSize, isVerbose)
%CHECK_PADSIZE  Padding needed to guarantee a minimum air gap around the mask.
%
%   [padPre, padPost, matrixSizePad] = check_padsize(mask)
%   [...] = check_padsize(mask, gapMinMM, isVerbose)
%
% The dipole convolution is performed with an FFT and is therefore
% CIRCULAR: susceptibility at one edge of the FOV produces field at
% the opposite edge. This matters far more for TFI than for
% local-field QSM, because the source here is the whole head against
% air (~9.4 ppm) rather than brain-internal variation (~0.1 ppm) -
% roughly two orders of magnitude more wraparound for the same
% geometry.
%
% Input
% -----
%   mask      : 3D ROI mask (logical or numeric)
%   gapMinMM  : minimum air gap required on EACH side, in mm.
%               Default 20.
%   isVerbose : print a per-dimension report. Default true.
%
% Output
% ------
%   padPre        : 1x3 voxels to prepend per dimension
%   padPost       : 1x3 voxels to append per dimension
%   matrixSizePad : resulting matrix size, factorable by 2/3/5/7
%
% Behaviour
%   1. Measure the existing gap on each side of the mask bounding box.
%   2. Add only what is missing, per side. A dimension that already
%      has enough air is left alone, so the in-plane dimensions of a
%      typical brain acquisition usually cost nothing and only the
%      through-slice direction grows.
%   3. Round the total up to the next size whose largest prime factor
%      is <= 7. An awkward size can cost more in FFT time than the
%      extra voxels buy, and the rounding is free because the extra
%      slices are all air.
%   4. Put the rounding slack on whichever side has less air, so an
%      off-centre object ends up closer to centred.
%
% Pad every spatial array identically. The fill values are NOT all
% zero:
%   data     -> 0
%   mask     -> false
%   weights  -> 0        (no fidelity where there is no signal)
%   MG       -> 1        (1 = no edge, so TV stays active there)
%   M2       -> false
%   P        -> this.Ps  (padded voxels are background; padding with
%                         1 would clamp them to |chi| <= ub and block
%                         exactly the distant sources being modelled)
%
% Verify sufficiency by convergence rather than by argument:
% reconstruct at gapMinMM and at 2*gapMinMM, and compare chi inside
% the ORIGINAL mask. When doubling the gap changes chi by well below
% the noise-equivalent susceptibility (~0.004 ppm at 0.5 Hz field
% noise, 3T), the padding is sufficient for that protocol.
%
% See also: PADARRAY

    if nargin < 3 || isempty(gapMinMM);  gapMinMM  = 20;   end
    if nargin < 4 || isempty(isVerbose); isVerbose = true; end

    mask       = mask > 0;
    matrixSize = size(mask, 1:3);

    if ~any(mask(:))
        error('gpuTFI:emptyMask','Mask is empty; cannot determine padding.');
    end

    gapMinVoxel = ceil(gapMinMM ./ voxelSize(:).');

    padPre  = zeros(1,3);
    padPost = zeros(1,3);
    extent  = zeros(1,3);
    gapPre  = zeros(1,3);
    gapPost = zeros(1,3);

    % ---- per-dimension gap deficit ------------------------------
    for d = 1:3

        other = setdiff(1:3, d);
        prof  = squeeze(any(any(mask, other(1)), other(2)));

        first = find(prof, 1, 'first');
        last  = find(prof, 1, 'last');

        extent(d)  = last - first + 1;
        gapPre(d)  = first - 1;
        gapPost(d) = matrixSize(d) - last;

        padPre(d)  = max(0, gapMinVoxel(d) - gapPre(d));
        padPost(d) = max(0, gapMinVoxel(d) - gapPost(d));

    end

    % ---- round up to a fast FFT size ----------------------------
    for d = 1:3

        nRaw  = matrixSize(d) + padPre(d) + padPost(d);
        extra = next_fast_fft_size(nRaw) - nRaw;

        if extra > 0
            % bias the slack toward the side with less air
            if (gapPre(d) + padPre(d)) <= (gapPost(d) + padPost(d))
                padPre(d)  = padPre(d)  + ceil(extra/2);
                padPost(d) = padPost(d) + floor(extra/2);
            else
                padPre(d)  = padPre(d)  + floor(extra/2);
                padPost(d) = padPost(d) + ceil(extra/2);
            end
        end

    end

    matrixSizePad = matrixSize + padPre + padPost;

    % ---- report -------------------------------------------------
    if isVerbose
        fprintf('----------------------------------------------------------\n');
        fprintf('Zero padding check (min gap %g mm per side)\n', gapMinMM);
        fprintf('----------------------------------------------------------\n');
        for d = 1:3
            fprintf(['dim %d: extent %3d vox | gap %5.1f/%5.1f mm | ' ...
                     'need %5.1f mm | pad %3d/%3d vox\n'], ...
                d, extent(d), ...
                gapPre(d)*voxelSize(d), gapPost(d)*voxelSize(d), ...
                gapMinVoxel(d)*voxelSize(d), padPre(d), padPost(d));
        end

        growth = prod(matrixSizePad) / prod(matrixSize);
        fprintf('----------------------------------------------------------\n');
        fprintf('matrix  : [%s] -> [%s]\n', ...
            num2str(matrixSize,'%d '), num2str(matrixSizePad,'%d '));
        fprintf('padPre  : [%s]   padPost : [%s]\n', ...
            num2str(padPre,'%d '), num2str(padPost,'%d '));
        fprintf('voxels  : %.2fx\n', growth);
        if growth > 4
            fprintf(['WARNING: memory grows %.1fx. Consider padding only the ' ...
                     'tightest dimension, or reducing gapMinMM.\n'], growth);
        end
        fprintf('----------------------------------------------------------\n');
    end

end

function n = next_fast_fft_size(n0)
% Smallest EVEN n >= n0 whose largest prime factor is <= 7.
%
% Evenness is required, not merely preferred. The dipole kernel is
% built on the grid -N/2 : N/2-1 and centred with fftshift, and both
% assume an even N:
%   - for odd N the grid is half-integer and contains no zero, so
%     k-space is sampled half a step off and there is no DC sample;
%   - fftshift equals ifftshift only for even N.
% An odd dimension therefore yields a silently wrong convolution
% operator. Guaranteeing even sizes here removes both failure modes
% for every caller of the kernel.
%
% Restricting to even sizes thins the candidate set, but only
% slightly: over 128-512 the worst-case padding overhead rises from
% 6.4% (any 7-smooth) to 8.5% (even 7-smooth), and is 0% whenever
% nRaw is already an even 7-smooth number, which covers the common
% acquisition matrices.

    n = max(round(n0), 2);
    n = n + mod(n, 2);                  % round up to even
    while max([1, factor(n)]) > 7
        n = n + 2;                      % step by 2 to stay even
    end

end