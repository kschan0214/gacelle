function assumeGPU(testCase)
% ASSUMEGPU Skip (not fail) the calling test if no GPU is available.
%
% GACELLE's askadam.m unconditionally calls gpuArray(...) with no CPU
% fallback, and no GPU-availability guard exists anywhere in the active
% (non-sandbox/deprecated) codebase. Every Tier-2 smoke-fit test must
% call this first so the suite degrades gracefully (reports as
% filtered/skipped) on machines and CI runners with no GPU, instead of
% hard-failing.
%
%   gacelletest.assumeGPU(testCase)
%
% Kwok-Shing Chan @ MGH

nGPU = 0;
try
    nGPU = gpuDeviceCount('available');
catch
    % Parallel Computing Toolbox not installed, or no CUDA driver -
    % treat as "no GPU available" rather than erroring the test.
end

testCase.assumeGreaterThan(nGPU, 0, ...
    'No GPU available - skipping GPU-dependent test.');

end
