classdef utils < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% 
% This is the class of all askadam realted functions
%
% Date created: 25 September 2024 
% Date modified: 
%
    properties (GetAccess = public, SetAccess = protected)

    end

    methods

    end

    methods(Static)

        % vectorise N-D image to 2D with the 1st dimension=spataial dimension and 2nd dimension=combine from 4th and onwards 
        function [data, mask_idx] = vectorise_NDto2D(data,mask)

            dims = size(data,[1 2 3]);

            if nargin < 2
                mask = ones(dims);
            end

             % vectorise data
            data        = reshape(data,prod(dims),prod(size(data,4:ndims(data))));
            mask_idx    = find(mask>0);
            data        = data(mask_idx,:);

            if ~isreal(data)
                data = cat(2,real(data),imag(data));
            end

        end

        % vectorise N-D image to 2D with the 1st dimension=spataial dimension and 2nd dimension=combine from 4th and onwards 
        function [data, mask_idx] = gpu_vectorise_NDto2D(data,mask)

            [data, mask_idx] = utils.vectorise_NDto2D(data,mask);

            % put data onto gpu
            data = gpuArray( single( data ));

        end

        % apply vectorise_NDto2D on all fields in the input structure
        function [data_struct] = vectorise_NDto2D_struct(data_struct,mask)

            % get fields
            fieldname = fieldnames(data_struct); 
            
            % loop all fields
            for km = 1:numel(fieldname)
                data_struct.(fieldname{km}) = utils.vectorise_NDto2D(data_struct.(fieldname{km}),mask); 
            end

        end

        % apply vectorise_NDto2D on all fields in the input structure
        function [data_struct] = gpu_vectorise_NDto2D_struct(data_struct,mask)

            % get fields
            fieldname = fieldnames(data_struct); 
            
            % loop all fields
            for km = 1:numel(fieldname)
                data_struct.(fieldname{km}) = utils.gpu_vectorise_NDto2D(data_struct.(fieldname{km}),mask); 
            end

        end

        % bring dlarray variable to cpu
        function data = dlarray2single(data)
            if isdlarray(data)
                data = extractdata(data);
            end
            if isgpuarray(data)
                data = gather(data);
            end
        end


        % this utility function to convert the MCMC posterior distribution into 4D/5D image
        function img = ND2image(dist,mask)
            
            imageDims = size(mask,1:3);
            extraDims = size(dist,2:ndims(dist));

            % find masked signal
            mask_idx            = find(mask>0);
            % reshape the input to an image         
            img                     = zeros(numel(mask),extraDims,'like',dist); 
            img(mask_idx,:,:,:,:,:) = dist; 
            img                     = reshape(img, [imageDims, extraDims]);
            
        end
        
        % make sure input vector is a row vector
        function vector = row_vector(vector)
            vector = reshape(vector, 1, []); 
        end
        
        % make sure data does not contain any NaN/Inf and update mask
        function [data,mask] = remove_img_naninf(data,mask)
        % Input
        % -------
        % data  : N-D image that may or may not contains NaN or Inf
        % mask  : 2D/3D mask
        %
        % Output
        % -------
        % data  : N-D image that is free from NaN or Inf
        % mask  : 2/3D mask that excludes NaN or Inf voxels
        %
            % mask sure no nan or inf
            Nvoxel_old              = numel(mask(mask>0));
            mask_nonnaninf          = and(~isnan(data) , ~isinf(data));
            data(mask_nonnaninf==0)  = 0;
            data(mask_nonnaninf==0)  = 0;
            for k = 4:ndims(data)
                mask_nonnaninf          = min(mask_nonnaninf,[],k);
            end
            mask                    = and(mask,mask_nonnaninf);
            Nvoxel_new              = numel(mask(mask>0));
            if Nvoxel_old ~= Nvoxel_new
                disp('The mask is updated due to the presence of NaN/Inf. Please make use of the output mask in your subseqeunt analysis.');
            end
        end

        % TODO: determine how the dataset will be divided based on vailable memory in GPU
        function [NSegment,maxSlice] = find_optimal_divide(mask,memoryFixPerVoxel,memoryDynamicPerVoxel)
        % Input
        % -----
        % mask                  : 3D signal mask
        % memoryFixPerVoxel     : memory usage 
        %
            % % get these number based on mdl fit
            % memoryFixPerVoxel       = 0.0013;
            % memoryDynamicPerVoxel   = 0.05;

            dims = size(mask,1:3);

            % GPU info
            gpu         = gpuDevice;    
            maxMemory   = floor(gpu.TotalMemory / 1024^3)*1024^3 / (1024^2);        % Mb

            % find max. memory required
            memoryRequiredFix       = memoryFixPerVoxel * prod(dims(1:3)) ;         % Mb
            memoryRequiredDynamic   = memoryDynamicPerVoxel * numel(mask(mask>0));  % Mb

            if maxMemory > (memoryRequiredFix + memoryRequiredDynamic)
                % if everything fit in GPU
                maxSlice = dims(3);
                NSegment = 1;
            else
                % if not then divide the data
                 NvolSliceMax= 0;
                for k = 1:dims(3)
                    tmp             = mask(:,:,k);
                    NvolSliceMax    = max(NvolSliceMax,numel(tmp(tmp>0)));
                end
                maxMemoryPerSlice = memoryDynamicPerVoxel * NvolSliceMax;
                maxSlice = floor((maxMemory - memoryRequiredFix)/maxMemoryPerSlice);
                NSegment = ceil(dims(3)/maxSlice);
            end

            fprintf('Data is divided into %d segments\n',NSegment);
        end

        % This function create a full out structure variable if the data is divided into multiple segments
        function out = restore_segment_structure(out,out_tmp,slice,ksegment)
        % Input
        % ---------
        % out       : askadam out structure final output 
        % out_tmp   : temporary out structure of each segment
        % slice     : slices where the segment belongs to
        % ksegment  : current segment number
        % 
            % reformat out structure
            fn1 = fieldnames(out_tmp);
            for kfn1 = 1:numel(fn1)
                fn2 = fieldnames(out_tmp.(fn1{kfn1}));
                for kfn2 = 1:numel(fn2)
                    if isscalar(out_tmp.(fn1{kfn1}).(fn2{kfn2})) % scalar value
                        out.(fn1{kfn1}).(fn2{kfn2})(ksegment) = out_tmp.(fn1{kfn1}).(fn2{kfn2});
                    else
                        % image result
                        out.(fn1{kfn1}).(fn2{kfn2})(:,:,slice,:,:) = out_tmp.(fn1{kfn1}).(fn2{kfn2});
                    end
                        
                end
            end
        end
        
        

    end

end