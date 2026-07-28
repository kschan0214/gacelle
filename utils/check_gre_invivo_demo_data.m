
download_dir = '~/Downloads'; % feel free to change this path

% dwi_invivo_dir = fullfile(download_dir,'ds006181');
gre_invivo_dir = '/autofs/space/virtuoso_001/users/kwokshing/project/gacelle/dsc';

cmd_txt = sprintf('aws s3 sync --no-sign-request s3://openneuro.org/ds006181 %s',gre_invivo_dir);

% if directory does not exist then download the data from OpenNeuro
if ~exist(gre_invivo_dir,'dir')
    system(cmd_txt);
end
