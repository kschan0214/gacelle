%% data_obj = setup_batch_create_data_obj(data, mask, numBatch, nElement, fieldName, data_obj)
%
% Input
% --------------
% data          : data to be store in batches, at least 2D, can have singleton dimension
% mask          : signal mask
% numBatch      : number of batches the data to be broken down (final)
% nElement      : number of elements in each batch
% fieldName     : field name to be store in data_obj, string
% data_obj      : (optional) previous created data_obj
% 
%
% Output
% --------------
% data_obj      : data_obj with data stored
%
% Kwok-shing Chan @ DCCN
% k.chan@donders.ru.nl
% Date created: 10 August 2021
% Date modified:
%
%
function data_obj = gpuMCRMWI_setup_batch_create_data_obj_slice(data, mask, fieldName, data_obj)

% initialise data obj
if nargin < 4 || isempty(data_obj)
    data_obj = [];
end

% get matrix size of the data
dim         = size(data);
if numel(dim) == 1
    dim = [dim, 1, 1];
elseif numel(dim) == 2
    dim = [dim, 1];
end

% vectorise image data for all voxels to 1st dimension
for kz = 1:dim(3)
    img_slice   = data(:,:,kz,:,:);
    mask_slice  = mask(:,:,kz);

    % find masked voxels
    ind = find(mask_slice>0);

    if ~isempty(ind)
        if nargin < 4
            data_obj(kz).isEmptySlice = false;
            data_obj(kz).mask         = mask_slice;
        end

        data_obj(kz).(fieldName) = img_slice;

    else
        if nargin < 4
            data_obj(kz).isEmptySlice   = true;
            data_obj(kz).mask           = mask_slice;
        end
        data_obj(kz).(fieldName)    = [];
    end

end

end