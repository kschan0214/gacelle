function addpath_gacelle(rootDir)
% ADDPATH_GACELLE Add GACELLE and its subfolders to path, excluding
% docs/ and any folder matching *sandbox*.
%
%   addpath_gacelle(rootDir)
%   addpath_gacelle()  % assumes current file's folder is rootDir
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 5 August 2026

if nargin < 1
    rootDir = fileparts(mfilename('fullpath'));
end

allPaths = strsplit(genpath(rootDir), pathsep);
allPaths(cellfun(@isempty, allPaths)) = [];

excludePattern = ["docs", "sandbox", "deprecated"];

keepMask = true(size(allPaths));
for k = 1:numel(allPaths)
    p = allPaths{k};
    for e = excludePattern
        % match as a path segment (docs) or anywhere in the name (sandbox)
        if e == "docs"
            hit = ~isempty(regexp(p, ['[\\/]', char(e), '([\\/]|$)'], 'once'));
        else
            hit = contains(p, e);
        end
        if hit
            keepMask(k) = false;
            break
        end
    end
end

addpath(allPaths{keepMask});

end