%% check_dwi_invivo_demo_data.m
% This script will check if the demo data exists.
% If not then download it 
% Kwok-Shing Chan 
% Date create: 6 Aug 2026
%

% if directory does not exist then download the data from OpenNeuro
if exist(dwi_invivo_dir,'dir')
    fprintf('Demo data exists in: %s\n',dwi_invivo_dir);
else
    fprintf('Demo data does not exist in: %s\n',dwi_invivo_dir);
    fprintf('Downloading the demo data to %s\n',dwi_invivo_dir);
    cmd_txt = sprintf('aws s3 sync --no-sign-request s3://openneuro.org/ds006181 %s',dwi_invivo_dir);
    system(cmd_txt);
    disp('Download is completed.');
end
