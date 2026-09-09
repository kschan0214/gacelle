%% check_gre_invivo_demo_data.m
% This script will check if the demo data exists.
% If not then download it
% Kwok-Shing Chan
% kchan2@mgh.harvard.edu
% Date created: 6 Aug 2026
% Date modified: 9 Sep 2026
%

% if the bids/ subfolder does not exist then download the data from Zenodo
if exist(fullfile(gre_invivo_dir,'bids'),'dir')
    fprintf('Demo data exists in: %s\n',gre_invivo_dir);
else
    fprintf('Demo data does not exist in: %s\n',gre_invivo_dir);
    fprintf('Downloading the demo data to %s\n',gre_invivo_dir);

    if ~exist(gre_invivo_dir,'dir'); mkdir(gre_invivo_dir); end

    % MATLAB's own file functions (exist/mkdir above) resolve a leading
    % '~' to the home directory, but gre_invivo_dir is still literally
    % stored as e.g. '~/Downloads/...'. Passed as-is into the bash -C
    % "..." argument below (double-quoted), that literal '~' is NOT
    % shell-expanded - tar would try to cd into a directory named '~'
    % that doesn't exist, fail immediately, and kill curl's write end of
    % the pipe ("Failed writing body"). Resolve it to an absolute path
    % here instead.
    gre_invivo_dir_abs = gre_invivo_dir;
    if strncmp(gre_invivo_dir_abs, '~', 1)
        gre_invivo_dir_abs = fullfile(getenv('HOME'), gre_invivo_dir_abs(2:end));
    end

    % The archive on Zenodo (10.5281/zenodo.22666992, ~750 MB) is
    % GACELLE's full data-sharing collection (dsc/), which bundles
    % several unrelated datasets. Only dsc/example_dataset_mcrmwi/bids
    % (the in vivo GRE/MCR-MWI example dataset the demos below expect at
    % fullfile(gre_invivo_dir,'bids')) is extracted here; curl is piped
    % straight into tar so the full archive is never written to disk.
    %
    % LD_LIBRARY_PATH="" and `bash -c ... pipefail` work around two
    % Linux/MATLAB system()-call gotchas: (1) MATLAB prepends its own
    % bundled (older) libcurl to LD_LIBRARY_PATH, which breaks the
    % system curl binary with "unknown option was passed in to libcurl"
    % on flags like -L/-f; (2) without pipefail, `curl | tar` silently
    % reports success (tar's exit code) even if curl itself failed and
    % piped nothing/an error page into tar.
    zenodo_url = 'https://zenodo.org/records/22666992/files/gacelle_dsc_20260908.tar.gz?download=1';
    cmd_txt    = sprintf(['bash -c ''set -o pipefail; LD_LIBRARY_PATH="" curl -fL "%s" | ' ...
                           'tar xz -C "%s" --strip-components=2 dsc/example_dataset_mcrmwi/bids'''], ...
                           zenodo_url, gre_invivo_dir_abs);
    status = system(cmd_txt);
    if status ~= 0
        error('check_gre_invivo_demo_data:downloadFailed', ...
            ['Failed to download/extract the demo data from Zenodo (exit status %d).\n' ...
             'Try running this command directly in a terminal (not inside MATLAB) to see the ' ...
             'actual curl/tar error:\n%s'], status, cmd_txt);
    end
    disp('Download is completed.');
end
