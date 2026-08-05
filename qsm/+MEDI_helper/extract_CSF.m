function maskCSF = extract_CSF(R2s, mask, voxelSize, flag_erode, thresh_R2s, opts)
%EXTRACT_CSF  Segment ventricular CSF as a zero-reference region for QSM.
%
%   maskCSF = extract_CSF(R2s, Mask, voxel_size)
%   maskCSF = extract_CSF(R2s, Mask, voxel_size, flag_erode, thresh_R2s)
%   maskCSF = extract_CSF(..., 'RadiusCentre', 30, ...)
%
% Strategy: CSF has low R2*, so threshold R2* and keep the connected
% components that intersect the largest low-R2* blobs near the centre of the
% ROI. The centre constraint is what separates ventricles from cortical CSF
% and from other low-R2* tissue at the periphery.
%
% Input
% -----
%   R2s         : R2* map [Hz], same size as Mask. May be empty, in which
%                 case [] is returned.
%   Mask        : ROI mask (brain). Logical or numeric.
%   voxel_size  : [dx dy dz] in mm
%   flag_erode  : erode the ROI with SMV before analysis (default true)
%   thresh_R2s  : low-R2* threshold [Hz] (default 5)
%
% Name-value
% ----------
%   RadiusCentre     : radius of the central sphere [mm]   (default 30)
%   NumCentreRegions : number of seed components to keep    (default 3)
%   ErodeRadius      : SMV radius for erosion [mm]          (default 10)
%   Connectivity     : bwconncomp connectivity              (default 6)
%
% Output
% ------
%   Mask_ROI_CSF : logical, same size as Mask
%
% NOTE: non-finite voxels in R2s are excluded, since NaN < thresh and
% Inf < thresh are both false. This is the desired behaviour but is silent -
% check nnz(~isfinite(R2s)) if the result looks unexpectedly small.
% adapted from MEDI toolbox

    arguments
        R2s
        mask
        voxelSize               (1,:) double
        flag_erode              (1,1) logical = true
        thresh_R2s              (1,1) double  = 5
        opts.RadiusCentre       (1,1) double = 30
        opts.NumCentreRegions   (1,1) double = 3
        opts.ErodeRadius        (1,1) double = 10
        opts.Connectivity       (1,1) double = 6
    end

    if isempty(R2s)
        maskCSF = [];
        return
    end

    mask        = mask > 0;
    matrixSize = size(mask);

    if ~isequal(size(R2s), matrixSize)
        error('extract_CSF:sizeMismatch', ...
              'R2s (%s) and Mask (%s) must be the same size.', ...
              mat2str(size(R2s)), mat2str(matrixSize));
    end
    if numel(voxelSize) < 3
        voxelSize = [voxelSize(:).' ones(1, 3-numel(voxelSize))];
    end

    nMask = nnz(mask);
    if nMask == 0
        warning('extract_CSF:emptyMask','Mask is empty; returning empty CSF ROI.');
        maskCSF = false(matrixSize);
        return
    end

    % ---- centroid of the ROI, in mm -------------------------------------
    % Computed from 1D marginals rather than full ndgrid arrays: the original
    % allocated three double volumes just to take three weighted means.
    x = (1:matrixSize(1)).' * voxelSize(1);
    y = (1:matrixSize(2))   * voxelSize(2);
    z = reshape((1:matrixSize(3)) * voxelSize(3), 1, 1, []);

    cx = sum(x .* sum(mask,[2 3])) / nMask;
    cy = sum(y .* sum(mask,[1 3])) / nMask;
    cz = sum(z .* sum(mask,[1 2])) / nMask;

    % ---- central sphere --------------------------------------------------
    % Implicit expansion; compare squared distances to avoid the sqrt.
    Mask_cen = (x-cx).^2 + (y-cy).^2 + (z-cz).^2 <= opts.RadiusCentre^2;

    % ---- optional erosion ------------------------------------------------
    % Kept in a separate variable: the original overwrote Mask, so the final
    % restriction silently used the eroded version.
    if flag_erode
        Mask_use = MEDI_helper.SMV(mask, matrixSize, voxelSize, opts.ErodeRadius) > 0.999;
    else
        Mask_use = mask;
    end

    lowR2s = R2s < thresh_R2s;      % non-finite -> false

    % ---- seed components near the centre ---------------------------------
    % Restricted to the ROI, unlike the original. For a brain-sized FOV the
    % 30 mm sphere sits well inside the mask so this is a no-op, but it makes
    % the function safe for small FOVs and non-brain applications.
    CC_cen = bwconncomp(lowR2s & Mask_cen & Mask_use, opts.Connectivity);

    if CC_cen.NumObjects == 0
        warning('extract_CSF:noSeed', ...
                ['No low-R2* component found within %g mm of the ROI centroid ' ...
                 '(threshold %g Hz). Returning empty CSF ROI.'], ...
                opts.RadiusCentre, thresh_R2s);
        maskCSF = false(matrixSize);
        return
    end

    % guard against fewer components than requested - the original indexed
    % idxs(1:3) unconditionally and errored when only 1 or 2 were found
    nSeed = min(opts.NumCentreRegions, CC_cen.NumObjects);

    numPixels  = cellfun(@numel, CC_cen.PixelIdxList);
    [~, order] = sort(numPixels, 'descend');

    seed              = false(matrixSize);
    seed(vertcat(CC_cen.PixelIdxList{order(1:nSeed)})) = true;

    % ---- components of the full ROI that touch a seed --------------------
    % labelmatrix replaces the original loop over every component, which was
    % the dominant cost when the threshold produced many small blobs.
    CC = bwconncomp(lowR2s & Mask_use, opts.Connectivity);
    L  = labelmatrix(CC);

    keep = unique(L(seed & L > 0));

    if isempty(keep)
        warning('extract_CSF:noOverlap', ...
                'Seed components did not intersect any ROI component; returning empty CSF ROI.');
        maskCSF = false(matrixSize);
        return
    end

    maskCSF = ismember(L, keep);   % already confined to Mask_use via L

end