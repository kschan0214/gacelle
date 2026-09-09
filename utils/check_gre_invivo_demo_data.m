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

    % The archive on Zenodo (10.5281/zenodo.22666992, ~750 MB) is
    % GACELLE's full data-sharing collection (dsc/), which bundles
    % several unrelated datasets. Only dsc/example_dataset_mcrmwi/bids
    % (the in vivo GRE/MCR-MWI example dataset the demos below expect at
    % fullfile(gre_invivo_dir,'bids')) is extracted here; curl is piped
    % straight into tar so the full archive is never written to disk.
    zenodo_url = 'https://zenodo.org/records/22666992/files/gacelle_dsc_20260908.tar.gz?download=1';
    cmd_txt    = sprintf(['curl -L "%s" | tar xz -C "%s" --strip-components=2 ' ...
                           'dsc/example_dataset_mcrmwi/bids'], zenodo_url, gre_invivo_dir);
    system(cmd_txt);
    disp('Download is completed.');
end
