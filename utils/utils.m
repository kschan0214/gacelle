classdef utils < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% 
% This is the class of all askadam realted functions
%
% Date created: 25 September 2024 
% Date modified: 
%

    properties (Constant)
        epsilon = 1e-8;
    end

    properties (GetAccess = public, SetAccess = protected)

    end

    methods

    end

    methods(Static)

        function [data_masked] = masking_ND2AD_preserve(data,mask)
        % this function concatenate the first 3 dimension of data and stored in the second dim, while preserving the 4th onward dim
        % data: [x,y,z,a,b,c] -> data_masked: [1,x*y*z,1,a,b,c]
            dims            = size(data,1:3);
            dims_nonspatial = size(data,4:ndims(data));

            if nargin < 2 || isempty(mask)
                mask = ones(dims);
            end

            % get mask index
            if numel(mask) == prod(dims)
                mask_idx    = find(mask>0);
            else
                mask_idx = mask;
            end
            
            data            = reshape(data,[1,prod(dims),1,dims_nonspatial]);
            data_masked     = data(1,mask_idx,1,:,:,:,:,:,:,:,:,:,:);

        end

        function data_struct = masking_ND2AD_preserve_struct(data_struct,mask)
            % get fields
            fieldname = fieldnames(data_struct); 

            % loop all fields
            for km = 1:numel(fieldname)
                dims = size(data_struct.(fieldname{km}),1:3);
                if all(dims == size(mask,1:3))
                    data_struct.(fieldname{km}) = utils.masking_ND2AD_preserve(data_struct.(fieldname{km}),mask); 
                end
            end

        end

        function [data] = undo_masking_ND2AD_preserve(data_masked,mask)

            dims            = size(mask,1:3);
            dims_nonspatial = size(data_masked,4:ndims(data_masked));

            if isempty(dims_nonspatial)
                data = utils.reshape_AD2ND(data_masked,mask);
            else

                mask_idx = find(mask>0);

                data             = zeros([prod(dims) dims_nonspatial], 'like', data_masked);
                data(mask_idx,:) = data_masked;
                data = reshape(data,[dims dims_nonspatial]);
                
            end

        end

        function data_struct = undo_masking_ND2AD_preserve_struct(data_struct,mask)
            % get fields
            fieldname = fieldnames(data_struct); 

            % loop all fields
            for km = 1:numel(fieldname)
                data_struct.(fieldname{km}) = utils.undo_masking_ND2AD_preserve(data_struct.(fieldname{km}),mask); 
            end
        end

        % this function reshape ND data into askAdam 2D (i.e.AD) input specific for this package, ie..[Nmeas,Nvoxel]
        function [data, mask_idx] = reshape_ND2AD(data,mask)

            [data, mask_idx] = utils.vectorise_NDto2D(data,mask);

            data = data.';

        end

        % this function reshape ND data stored in a structure array into askAdam 2D (i.e.AD) input specific for this package, ie..[Nmeas,Nvoxel]
        function data_struct = reshape_ND2AD_struct(data_struct,mask)

            % get fields
            fieldname = fieldnames(data_struct); 
            
            % loop all fields
            for km = 1:numel(fieldname)
                data_struct.(fieldname{km}) = utils.reshape_ND2AD(data_struct.(fieldname{km}),mask); 
            end

        end

        % this function reshape ND data stored in a structure array into askAdam 2D (i.e.AD) input specific for this package, ie..[Nmeas,Nvoxel]
        function data_struct = gpu_reshape_ND2AD_struct(data_struct,mask)

            % get fields
            fieldname = fieldnames(data_struct); 
            
            % loop all fields
            for km = 1:numel(fieldname)
                data_struct.(fieldname{km}) = gpuArray(single( utils.reshape_ND2AD(data_struct.(fieldname{km}),mask) )); 
            end

        end

        % undo reshape_ND2AD
        function data = reshape_AD2ND(data,mask)

            data = utils.reshape_ND2image(data.',mask);

        end

        % undo reshape_ND2AD_struct
        function data_struct = reshape_AD2ND_struct(data_struct,mask)

            % get fields
            fieldname = fieldnames(data_struct);

            % loop all fields
            for km = 1:numel(fieldname)
                data_struct.(fieldname{km}) = utils.reshape_AD2ND(data_struct.(fieldname{km}),mask); 
            end
            

        end

        % reshape N-D image to 2D with the 1st dimension=spataial dimension and 2nd dimension=combine from 4th and onwards 
        function [data, mask_idx] = vectorise_NDto2D(data,mask)
        % mask can be (1-3)D or 1-D index 

            dims = size(data,[1 2 3]);

            if nargin < 2 || isempty(mask)
                mask = ones(dims);
            end

             % vectorise data
            data        = reshape(data,prod(dims),prod(size(data,4:ndims(data))));
            % get mask index
            if numel(mask) == prod(dims)
                mask_idx    = find(mask>0);
            else
                mask_idx = mask;
            end
            data        = data(mask_idx,:);

            if ~isreal(data)
                data = cat(2,real(data),imag(data));
            end

        end

        function [dataND] = vectorise_2DtoND(data2D,mask)
            dims = size(mask,1:3);

            mask_idx = find(mask>0);

            dataND = zeros(numel(mask),size(data2D,2));
            dataND(mask_idx,:) = data2D;

            dataND = reshape(dataND,[dims size(data2D,2)]);

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
        function img = reshape_ND2image(dist,mask)
            
            imageDims = size(mask,1:3);
            extraDims = size(dist,2:ndims(dist));

            % find masked signal
            mask_idx            = find(mask>0);
            % reshape the input to an image         
            img                     = zeros(numel(mask),extraDims,'like',dist); 
            img(mask_idx,:,:,:,:,:) = dist; 
            img                     = reshape(img, [imageDims, extraDims]);
            
        end

        function data_struct = reshape_2DinputtoND_struct(data_struct,mask)

            % get fields
            fieldname = fieldnames(data_struct);

            % loop all fields
            for km = 1:numel(fieldname)
                data_struct.(fieldname{km}) = utils.reshape_ND2image(data_struct.(fieldname{km}).',mask); 
            end

        end

        %  % this utility function to convert the MCMC posterior distribution into 4D/5D image
        % function data_struct = reshape_ND2image_struct(data_struct,mask)
        % 
        %     % get fields
        %     fieldname = fieldnames(data_struct); 
        % 
        %     % loop all fields
        %     for km = 1:numel(fieldname)
        %         data_struct.(fieldname{km}) = utils.reshape_ND2image(data_struct.(fieldname{km}),mask); 
        %     end
        % 
        % end
        
        % make sure input vector is a row vector
        function vector = row_vector(vector)
            vector = reshape(vector, 1, []); 
        end

        function [data, mask] = set_nan_inf_zero(data)
            mask = or(isnan(data), isinf(data));
            data(mask)  = 0;
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
            [data, masknaninf]      = utils.set_nan_inf_zero(data);
            mask_nonnaninf          = ~masknaninf;
            % mask_nonnaninf          = and(~isnan(data) , ~isinf(data));
            % data(mask_nonnaninf==0)  = 0;
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
            if NSegment ~= 1
                fprintf('Data is divided into %d segments\n',NSegment);
            end
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
                        try
                            if ksegment == 1
                                out.(fn1{kfn1}).(fn2{kfn2}) = out_tmp.(fn1{kfn1}).(fn2{kfn2});
                            else
                                out.(fn1{kfn1}).(fn2{kfn2})(:,:,slice,:,:) = out_tmp.(fn1{kfn1}).(fn2{kfn2});
                            end
                        catch
                            if ksegment == 1
                                out.(fn1{kfn1}).(fn2{kfn2}) = out_tmp.(fn1{kfn1}).(fn2{kfn2});
                            else
                                out.(fn1{kfn1}).(fn2{kfn2}) = cat(1,out.(fn1{kfn1}).(fn2{kfn2}) ,out_tmp.(fn1{kfn1}).(fn2{kfn2}));
                            end
                        end
                    end
                        
                end
            end
        end
        
         % initialise parameters
         function parameters = initialise_x0(dims,modelParams,startingPoint)
            
            for k = 1:numel(modelParams)
                parameters.(modelParams{k}) = ones(dims,'single') *startingPoint(k);
            end

         end

         function txt = logical2string(trueFalse)
             trueFalse = logical(trueFalse);
             if trueFalse
                 txt = 'true';
             else
                 txt = 'false';
             end
         end
    end

end