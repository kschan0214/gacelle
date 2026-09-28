function G = grid3_refine(logpostFun, ranges, nFine, nCoarse, thresh, nCheck)
% GRID3_REFINE Brute-force 3D grid posterior with one refinement pass (double precision, GPU).
%
%   G = grid3_refine(logpostFun, ranges, nFine, nCoarse, thresh, nCheck)
%
% logpostFun : @(A,B,C) unnormalised log posterior; A [nA,1,1], B [1,nB,1],
%              C [1,1,nC] (gpuArray double) -> [nA,nB,nC]; -Inf outside the prior box
% ranges     : [3 x 2] prior box (or a range that contains all posterior mass)
% nFine      : nodes per axis of the refined grid
% nCoarse    : nodes per axis of the coarse grid over 'ranges'
% thresh     : the refined box is the bounding box of the coarse nodes with
%              logpost > max - thresh, expanded by 2 coarse nodes, clipped to 'ranges'
% nCheck     : nodes per axis of a second refined grid (same box), used to
%              estimate the discretisation error of the quantiles
%
% Output G
%   .axes{k}      : refined nodes of axis k
%   .marg{k}      : unnormalised marginal density at the refined nodes (trapezoid over the other axes)
%   .axesCheck{k}, .margCheck{k} : same on the nCheck grid
%   .box          : [3 x 2] refined box
%   .edgeMass     : max over axes of the normalised marginal density at the refined box ends times
%                   the node spacing, for ends that are NOT prior-box ends (should be ~0)
%   .logp, .w     : refined log posterior (max 0) and trapezoid cell weights (host double), for resampling
%
% The trapezoid rule on a node grid that resolves the posterior (spacing ~0.1
% marginal SD here) is accurate far beyond Monte Carlo error; nCheck gives an
% empirical estimate of the remaining error.
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%
if nargin < 6 || isempty(nCheck); nCheck = round(2*nFine/3); end

% coarse pass over the full range
[lpC, axC] = evaluate(logpostFun, ranges, nCoarse);
isIn = lpC > max(lpC(:)) - thresh;
box  = ranges;
for k = 1:3
    other = setdiff(1:3, k);
    anyK  = squeeze(any(any(permute(isIn, [k other]), 3), 2));
    lo    = max(find(anyK, 1, 'first') - 2, 1);
    hi    = min(find(anyK, 1, 'last')  + 2, nCoarse);
    box(k,:) = [axC{k}(lo) axC{k}(hi)];
end
G.box = box;

% refined grids
[G.logp, G.axes, G.marg, G.w] = refined(logpostFun, box, nFine);
[~, G.axesCheck, G.margCheck] = refined(logpostFun, box, nCheck);

% mass near refined-box ends that are interior to the prior box
edge = 0;
for k = 1:3
    x = G.axes{k}; p = G.marg{k} / trapz(x, G.marg{k}); h = x(2)-x(1);
    if box(k,1) > ranges(k,1); edge = max(edge, p(1)*h);   end
    if box(k,2) < ranges(k,2); edge = max(edge, p(end)*h); end
end
G.edgeMass = edge;
end

function [lp, ax, marg, w] = refined(logpostFun, box, n)
[lp, ax] = evaluate(logpostFun, box, n);
lp  = lp - max(lp(:));
p   = exp(lp);
tw  = cell(1,3);
for k = 1:3
    h = ax{k}(2) - ax{k}(1);
    tw{k} = h * [0.5; ones(n-2,1); 0.5];
end
w    = tw{1} .* reshape(tw{2},1,[]) .* reshape(tw{3},1,1,[]);
marg = cell(1,3);
marg{1} = squeeze(sum(sum(p .* reshape(tw{2},1,[]) .* reshape(tw{3},1,1,[]), 2), 3));
marg{2} = squeeze(sum(sum(p .* tw{1} .* reshape(tw{3},1,1,[]), 1), 3));
marg{3} = squeeze(sum(sum(p .* tw{1} .* reshape(tw{2},1,[]), 1), 2));
for k = 1:3; marg{k} = marg{k}(:); end
end

function [lp, ax] = evaluate(logpostFun, box, n)
ax = cell(1,3);
for k = 1:3; ax{k} = linspace(box(k,1), box(k,2), n).'; end
B  = gpuArray(reshape(ax{2}, 1, []));
C  = gpuArray(reshape(ax{3}, 1, 1, []));
lp = zeros(n, n, n);
slab = 10;
for i = 1:slab:n
    idx = i:min(i+slab-1, n);
    lp(idx,:,:) = gather(logpostFun(gpuArray(ax{1}(idx)), B, C));
end
end
