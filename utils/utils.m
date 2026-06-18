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

        function data = vectorise(data)
            data = data(:);
        end

        function [data_masked] = masking_ND2GD_preserve(data,mask)
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

        function data_struct = masking_ND2GD_preserve_struct(data_struct,mask)
            % get fields
            fieldname = fieldnames(data_struct); 

            % loop all fields
            for km = 1:numel(fieldname)
                dims = size(data_struct.(fieldname{km}),1:3);
                if all(dims == size(mask,1:3))
                    data_struct.(fieldname{km}) = utils.masking_ND2GD_preserve(data_struct.(fieldname{km}),mask); 
                end
            end

        end

        function [data] = undo_masking_ND2GD_preserve(data_masked,mask)

            dims            = size(mask,1:3);
            dims_nonspatial = size(data_masked,4:ndims(data_masked));

            if isempty(dims_nonspatial)
                data = utils.reshape_GD2ND(data_masked,mask);
            else

                mask_idx = find(mask>0);

                data             = zeros([prod(dims) dims_nonspatial], 'like', data_masked);
                data(mask_idx,:) = data_masked;
                data = reshape(data,[dims dims_nonspatial]);
                
            end

        end

        function data_struct = undo_masking_ND2GD_preserve_struct(data_struct,mask)
            % get fields
            fieldname = fieldnames(data_struct); 

            % loop all fields
            for km = 1:numel(fieldname)
                if ~isscalar(data_struct.(fieldname{km})) && ~isscalar(mask)
                    data_struct.(fieldname{km}) = utils.undo_masking_ND2GD_preserve(data_struct.(fieldname{km}),mask); 
                end
            end
        end

        % this function reshape ND data into GACELLE 2D (i.e.GD) input specific for this package, ie..[Nmeas,Nvoxel]
        function [data, mask_idx] = reshape_ND2GD(data,mask)

            [data, mask_idx] = utils.vectorise_NDto2D(data,mask);

            data = data.';

        end

        % this function reshape ND data stored in a structure array into askAdam 2D (i.e.AD) input specific for this package, ie..[Nmeas,Nvoxel]
        function data_struct = reshape_ND2GD_struct(data_struct,mask)

            % get fields
            fieldname = fieldnames(data_struct); 
            
            % loop all fields
            for km = 1:numel(fieldname)
                if ~isscalar(data_struct.(fieldname{km}))
                    data_struct.(fieldname{km}) = utils.reshape_ND2GD(data_struct.(fieldname{km}),mask); 
                end
            end

        end

        % this function reshape ND data stored in a structure array into askAdam 2D (i.e.AD) input specific for this package, ie..[Nmeas,Nvoxel]
        function data_struct = gpu_reshape_ND2GD_struct(data_struct,mask)

            % get fields
            fieldname = fieldnames(data_struct); 
            
            % loop all fields
            for km = 1:numel(fieldname)
                data_struct.(fieldname{km}) = gpuArray(single( utils.reshape_ND2GD(data_struct.(fieldname{km}),mask) )); 
            end

        end

        % undo reshape_ND2GD
        function data = reshape_GD2ND(data,mask)

            data = utils.reshape_ND2image(data.',mask);

        end

        % undo reshape_ND2GD_struct
        function data_struct = reshape_GD2ND_struct(data_struct,mask)

            % get fields
            fieldname = fieldnames(data_struct);

            % loop all fields
            for km = 1:numel(fieldname)
                data_struct.(fieldname{km}) = utils.reshape_GD2ND(data_struct.(fieldname{km}),mask); 
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
            
            if ~isscalar(dist)
                imageDims = size(mask,1:3);
                extraDims = size(dist,2:ndims(dist));
    
                % find masked signal
                mask_idx            = find(mask>0);
                % reshape the input to an image         
                img                     = zeros(numel(mask),extraDims,'like',dist); 
                img(mask_idx,:,:,:,:,:) = dist; 
                img                     = reshape(img, [imageDims, extraDims]);
            else
                img = dist;
            end
            
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

        %%%%%% Memory management

        function [nvidiaPID, logFile, matlabPID] = start_probe_logging()
            matlabPID = feature('getpid');
            logFile   = strcat(tempname, '_gacelle_probe.csv');
            
            % Log total GPU memory AND per-process breakdown
            % Two columns: total_used, matlab_process_used
            % We poll total (high time resolution) and correct for non-MATLAB usage
            cmd = sprintf(['nvidia-smi --query-gpu=memory.used ' ...
                           '--format=csv,noheader,nounits -lms 5 > %s &'], logFile);
            system(cmd);
            [~, pidStr] = system('pgrep -n nvidia-smi');
            nvidiaPID   = strtrim(pidStr);
        end
        
        function otherMem_MB = get_other_process_memory(matlabPID)
        % Query memory used by all GPU processes EXCEPT MATLAB.
        % This is used to correct the total-GPU readings.
            [status, cmdout] = system('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits');
            if status ~= 0 || isempty(strtrim(cmdout))
                otherMem_MB = 0;
                return
            end
            
            lines       = strtrim(splitlines(strtrim(cmdout)));
            otherMem_MB = 0;
            for k = 1:numel(lines)
                if isempty(lines{k}); continue; end
                parts = strsplit(lines{k}, ',');
                if numel(parts) < 2; continue; end
                pid = str2double(strtrim(parts{1}));
                mem = str2double(strtrim(parts{2}));
                if pid ~= matlabPID && ~isnan(mem)
                    otherMem_MB = otherMem_MB + mem;
                end
            end
        end

        function absolutePeak_MiB = read_absolute_peak_from_log(logFile)
        % Return the absolute maximum GPU memory used during the probe window,
        % after discarding early startup samples.
        % Caller is responsible for subtracting other-process memory.
        
            try
                attempts = 0;
                while ~isfile(logFile) && attempts < 20
                    pause(0.05); attempts = attempts + 1;
                end
        
                T    = readtable(logFile, 'FileType','text', 'Delimiter',',', ...
                                 'VariableNamingRule','preserve');
                vals = T{:,1};
                if ~isnumeric(vals)
                    vals = str2double(erase(string(vals), ' MiB'));
                end
                vals = vals(~isnan(vals));
        
                if numel(vals) < 5
                    warning('GACELLE:memoryLog', 'Too few samples (%d) in %s', numel(vals), logFile);
                    absolutePeak_MiB = max(vals);
                    return
                end
        
                % Discard first 5% - nvidia-smi startup + fit initialisation noise
                nDiscard         = max(2, round(0.05 * numel(vals)));
                vals             = vals(nDiscard+1:end);
                absolutePeak_MiB = max(vals);
        
                fprintf('    [mem log] samples=%d, discarded=%d, peak=%.0f MiB\n', ...
                    numel(vals)+nDiscard, nDiscard, absolutePeak_MiB);
        
            catch ME
                warning('GACELLE:memoryLog', 'Could not read %s: %s', logFile, ME.message);
                g                = gpuDevice;
                absolutePeak_MiB = (g.TotalMemory - g.FreeMemory) / 1024^2;
            end
        end

        % Add get_available_vram to utils.m
        function memAvail_MiB = get_available_vram(gpuIndex)
            if nargin < 1; gpuIndex = 0; end
            [status, cmdout] = system(sprintf(...
                'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=%d', gpuIndex));
            if status == 0
                lines        = strtrim(splitlines(strtrim(cmdout)));
                memAvail_MiB  = str2double(lines{1});
            else
                warning('GACELLE:memoryQuery', ...
                    'nvidia-smi unavailable; falling back to gpu.AvailableMemory. May be inaccurate on shared HPC nodes.');
                g            = gpuDevice;
                memAvail_MiB  = g.AvailableMemory / 1024^2;
            end
        end

        function [seg,NSegment] = find_optimal_segment_3D(modelObj, data, mask, fitting, varargin)
        % modelObj : the model object (e.g. gpuMCRMWI, gpuJointR1R2starMapping)
        %            must implement: fit_probe, slice_segment, fit_segment, postprocess_segments
        %
        % fitting.segmentOverlap (default 0)  : number of halo slices added on each
        %                                        *internal* segment boundary, to give
        %                                        3D-coupled regularisers (e.g. 3D TV)
        %                                        correct neighbour information across
        %                                        segment cuts. Has no effect when
        %                                        NSegment == 1. Default 0 reproduces
        %                                        legacy (no-halo) behaviour exactly.
        % fitting.NSegmentUser (default [])   : user-requested minimum segment count.
        %                                        Treated as a FLOOR, not an override:
        %                                        NSegment = max(NSegmentUser, memoryRequired).
        %                                        This guarantees the memory logic can
        %                                        only ever add segments, never remove
        %                                        the safety margin it computed, so this
        %                                        option cannot by itself cause an OOM.
        %
        % Output
        % seg : struct array, one entry per segment, with fields:
        %         .owned  - global slice indices this segment is responsible for
        %                   (used to write results back; disjoint across segments,
        %                   and exactly partitions 1:size(mask,3))
        %         .fit    - global slice indices actually extracted/fitted, i.e.
        %                   .owned padded with up to fitting.segmentOverlap halo
        %                   slices on internal faces only (never past the true
        %                   volume boundary)
        %         .local  - position of .owned within .fit (i.e. .owned - .fit(1) + 1),
        %                   precomputed here so callers never re-derive this arithmetic
        %       When NSegment == 1, seg(1).owned == seg(1).fit == 1:size(mask,3) and
        %       seg(1).local == seg(1).owned, identical to today's single-segment case.

        % if nargin < 4; safetyFactor = 1; end
            safetyFactor = 1;

            % --- new opt-in options, both default to legacy behaviour ---
            if ~isfield(fitting,'segmentOverlap') || isempty(fitting.segmentOverlap)
                fitting.segmentOverlap = 0;
            end
            if ~isfield(fitting,'NSegmentUser')
                fitting.NSegmentUser = [];
            end
            h            = fitting.segmentOverlap;
            NSegmentUser = fitting.NSegmentUser;
            if isempty(NSegmentUser) || NSegmentUser < 1
                NSegmentUser = 1;
            else
                NSegmentUser = round(NSegmentUser);
            end

            gpu = gpuDevice; reset(gpu);

            Nvoxel      = nnz(mask);
            probeMin    = 100;
            probeMax    = min(round(Nvoxel*0.1),1e5);
            probeSize = [probeMin probeMax];
            if probeMax <= probeMin
                fitting.autoMemManage = 0;  % size too small
            end

            % NOTE: the NSegmentUser floor is honoured even when autoMemManage is
            % off or skipped (small data) below - a user asking for N segments for
            % a non-memory reason (e.g. seam validation) should not be silently
            % ignored just because the memory probe didn't run. If this is not what
            % you want, gate the NSegmentUser>1 branches below on fitting.autoMemManage.

            if fitting.autoMemManage && Nvoxel > max(probeSize)
        
                fprintf('Checking GPU memory requirements...\n');

                matlabPID     = feature('getpid');
                matlabPeak_MB = nan(1, numel(probeSize));   % absolute MATLAB peak, not delta

                for kp = 1:numel(probeSize)
                    mask_probe                  = zeros(size(mask));
                    mask_probe(1:probeSize(kp)) = 1;
                    fitting_probe               = fitting;
                    fitting_probe.iteration     = 0;
                    fitting_probe.start         = 'default';
                    % fitting_probe still carries segmentOverlap/NSegmentUser, but
                    % this is harmless: modelObj.fit() never reads those fields,
                    % only find_optimal_segment_3D (this function) does, and fit()
                    % is not re-entered through this function during probing.
                
                    logFile   = strcat(tempname, '_gacelle_probe.csv');
                    cmd       = sprintf('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 5 > %s &', logFile);
                    system(cmd);
                    [~, pidStr] = system('pgrep -n nvidia-smi');
                    nvidiaPID   = strtrim(pidStr);
                
                    % Other-process memory snapshot before fit
                    otherMem_before_MB = utils.get_other_process_memory(matlabPID);
                
                    try
                        [~, ~] = evalc('modelObj.fit(data, mask_probe, fitting_probe, varargin{:})');
                    catch ME
                        system(sprintf('kill %s 2>/dev/null', nvidiaPID));
                        if isfile(logFile); delete(logFile); end
                        if contains(ME.message, 'out of memory', 'IgnoreCase', true)
                            warning('GACELLE:probeOOM', 'OOM at probe size %d', probeSize(kp));
                            continue
                        else
                            rethrow(ME);
                        end
                    end
                
                    system(sprintf('kill %s 2>/dev/null', nvidiaPID));
                    pause(0.1);
                
                    % Other-process memory snapshot after fit
                    otherMem_after_MB  = utils.get_other_process_memory(matlabPID);
                
                    % Best estimate of other-process memory during probe peak:
                    % use max(before, after) - conservative, assumes worst-case contamination
                    otherMem_peak_MB   = max(otherMem_before_MB, otherMem_after_MB);
                
                    % Absolute peak total GPU usage from log (includes other processes)
                    totalPeak_MB       = utils.read_absolute_peak_from_log(logFile);
                
                    % MATLAB-only absolute peak
                    matlabPeak_MB(kp)  = max(0, totalPeak_MB - otherMem_peak_MB);
                
                    fprintf('  Probe %d/%d (N=%4d voxels): MATLAB peak = %.0f MiB (total=%.0f, other=%.0f)\n', ...
                        kp, numel(probeSize), probeSize(kp), ...
                        matlabPeak_MB(kp), totalPeak_MB, otherMem_peak_MB);
                
                    if isfile(logFile); delete(logFile); end
                end
                
                % Fit absolute MATLAB peak vs probe size
                % mem_matlab_peak = slope * Nvoxels + intercept
                validIdx  = ~isnan(matlabPeak_MB);
                coef      = polyfit(probeSize(validIdx), matlabPeak_MB(validIdx), 1);
                slope     = coef(1);      % MB per voxel (all costs: baseline + autodiff)
                intercept = coef(2);      % MB fixed MATLAB overhead

                % Predict full-problem MATLAB peak memory
                memPred_MB = polyval(coef, Nvoxel);
                
                % Compare against available VRAM from nvidia-smi (other-process-aware)
                memAvail_MB   = utils.get_available_vram();
                memBudget_MB  = memAvail_MB * safetyFactor;
                
                fprintf('Memory prediction:\n');
                fprintf('  Predicted MATLAB peak : %.0f MB\n', memPred_MB);
                fprintf('  Available VRAM (smi)  : %.0f MB\n', memAvail_MB);
                fprintf('  Budget (%.0f%%)         : %.0f MB\n', safetyFactor*100, memBudget_MB);
        
                if memAvail_MB < memPred_MB || NSegmentUser > 1
                    if memAvail_MB < memPred_MB
                        warning('GACELLE:memoryWarning', ...
                                'Predicted memory (%.2f MB) exceeds 100%% of available GPU memory (%.2f MB). Segmenting data.', ...
                                memPred_MB, memAvail_MB);
                    end

                    % Solve for max voxels per FIT segment (owned + halo):
                    % slope * NvoxPerSegFit + intercept <= memAvail
                    NvoxPerSegFit = floor((memAvail_MB - intercept) / slope);

                    if NvoxPerSegFit <= 0
                        error('GACELLE:memoryError', ...
                              'Insufficient GPU memory even for a single segment. Predicted fixed overhead (%.2f GB) already exceeds available memory (%.2f GB).', ...
                              intercept, memAvail_MB);
                    end

                    % Charge the halo against the budget BEFORE sizing owned ranges,
                    % so a fit segment (owned+halo) never exceeds NvoxPerSegFit.
                    % Use mean slice density as the per-slice voxel cost estimate,
                    % and the worst case of 2 halo faces (interior segment) so the
                    % owned-target is conservative for every segment, not just edges.
                    sliceDensity_  = squeeze(sum(mask, [1 2]));
                    meanSliceVox   = mean(sliceDensity_(sliceDensity_>0));
                    if isempty(meanSliceVox) || isnan(meanSliceVox); meanSliceVox = 0; end
                    haloVoxCost    = 2 * h * meanSliceVox;
                    NvoxPerSegOwned = max(1, NvoxPerSegFit - haloVoxCost);

                    if NvoxPerSegFit <= haloVoxCost
                        warning('GACELLE:haloBudget', ...
                            ['Overlap (h=%d slices) consumes the entire per-segment memory budget. ' ...
                             'Falling back to NvoxPerSegOwned=1 voxel/segment target; consider reducing ' ...
                             'fitting.segmentOverlap or using a GPU with more VRAM.'], h);
                    end

                    % Respect the user-requested minimum segment count as a FLOOR:
                    % NSegment can only be pushed UP from the memory-derived value,
                    % never down, so this option cannot cause an OOM by itself.
                    NSegmentMemory  = max(1, ceil(sum(sliceDensity_) / NvoxPerSegOwned));
                    NSegmentTarget  = max(NSegmentMemory, NSegmentUser);

                    % Build density-balanced OWNED slice boundaries so each segment
                    % has approximately equal owned voxel counts, honouring NSegmentTarget
                    ownedBoundaries = utils.build_balanced_boundaries(mask, NvoxPerSegOwned, NSegmentTarget);
                    NSegment        = numel(ownedBoundaries);
                    fprintf('Data divided into %d segments (target %d owned voxels/segment, halo=%d slices)\n', ...
                        NSegment, round(NvoxPerSegOwned), h);
                    if NSegment > 1
                        fprintf('The estimation may not be exactly the same as 1 segment.\n');
                    end

                    seg = utils.expand_segments_with_halo(ownedBoundaries, h, size(mask,3));

                else
                    fprintf('Full data fits in GPU memory (predicted %.2f GB / available %.2f GB)\n', ...
                        memPred_MB, memAvail_MB/safetyFactor);
                    seg      = struct('owned', {1:size(mask,3)}, 'fit', {1:size(mask,3)}, 'local', {1:size(mask,3)});
                    NSegment = 1;
                end
            
            else
                % Probe skipped (autoMemManage off, or too little data to probe
                % reliably). NSegmentUser is still honoured as a floor here - see
                % NOTE above - but with no memory information, "owned" boundaries
                % are simply equal-thickness slabs rather than density-balanced.
                if NSegmentUser > 1
                    ownedBoundaries = utils.build_balanced_boundaries(mask, ceil(nnz(mask)/NSegmentUser), NSegmentUser);
                    NSegment        = numel(ownedBoundaries);
                    seg             = utils.expand_segments_with_halo(ownedBoundaries, h, size(mask,3));
                else
                    seg      = struct('owned', {1:size(mask,3)}, 'fit', {1:size(mask,3)}, 'local', {1:size(mask,3)});
                    NSegment = 1;
                end
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

        function boundaries = build_balanced_boundaries(mask, NvoxPerSeg, NSegmentMin)
        % Divide slices into segments with approximately equal voxel counts,
        % where each segment stays <= NvoxPerSeg masked voxels.
        %
        % NSegmentMin (optional, default 1): floor on the number of segments,
        % e.g. from a user-requested minimum (fitting.NSegmentUser). Does not
        % reduce memory safety - it can only ever increase segment count, which
        % can only ever decrease per-segment voxel load.
        %
        % Strategy: compute cumulative voxel count across slices, then find
        % slice indices where cumulative count crosses multiples of the target
        % per-segment count.

            if nargin < 3 || isempty(NSegmentMin); NSegmentMin = 1; end
        
            sliceDensity = squeeze(sum(mask, [1 2]));   % [dims3 x 1]
            dims3        = size(mask, 3);
            totalVox     = sum(sliceDensity);
        
            % How many segments do we actually need?
            NSegment     = max(ceil(totalVox / NvoxPerSeg), NSegmentMin);
            targetPerSeg = totalVox / NSegment;          % equal voxel target per segment
        
            % Walk cumulative sum and cut when we cross each target boundary
            cumVox     = cumsum(sliceDensity);
            boundaries = cell(NSegment, 1);
            segStart   = 1;
        
            for ks = 1:NSegment
                if ks < NSegment
                    % Find the slice where cumulative voxels first reaches this
                    % segment's target, choosing the cut that minimises imbalance
                    target    = ks * targetPerSeg;
                    % Last slice before we exceed target
                    below     = find(cumVox < target, 1, 'last');
                    % First slice at or above target  
                    above     = find(cumVox >= target, 1, 'first');
        
                    if isempty(below)
                        % All slices exceed target - take just one slice
                        segEnd = segStart;
                    elseif isempty(above)
                        segEnd = dims3;
                    else
                        % Pick whichever cut gives closer to targetPerSeg voxels
                        vox_if_below = cumVox(below) - (ks-1)*targetPerSeg;
                        vox_if_above = cumVox(above) - (ks-1)*targetPerSeg;
                        if abs(vox_if_below - targetPerSeg) <= abs(vox_if_above - targetPerSeg)
                            segEnd = below;
                        else
                            segEnd = above;
                        end
                    end
                    % Guard: ensure we always make forward progress
                    segEnd = max(segEnd, segStart);
                else
                    % Last segment takes whatever remains
                    segEnd = dims3;
                end
        
                boundaries{ks} = segStart:segEnd;
                segStart       = segEnd + 1;
            end
        
            % Remove any empty segments (can occur with very sparse masks)
            boundaries = boundaries(~cellfun(@isempty, boundaries));
        
            % Diagnostic: report actual voxel counts per segment
            fprintf('Segment voxel counts (target = %d per segment):\n', round(targetPerSeg));
            for ks = 1:numel(boundaries)
                slices   = boundaries{ks};
                segVox   = sum(sliceDensity(slices));
                fprintf('  Segment %d: slices %d-%d, %d voxels (%.1f%% of target)\n', ...
                    ks, slices(1), slices(end), segVox, 100*segVox/targetPerSeg);
            end
        end

        function seg = expand_segments_with_halo(ownedBoundaries, h, dims3)
        % Expand a cell array of disjoint, contiguous OWNED slice ranges (as
        % returned by build_balanced_boundaries) into a struct array carrying
        % owned/fit/local geometry for halo-aware segmented fitting.
        %
        % Input
        %   ownedBoundaries : cell array of slice-index vectors, one per segment,
        %                     contiguous and exactly partitioning 1:dims3
        %   h               : halo width in slices, applied on INTERNAL faces only
        %                     (the true volume boundary, slice 1 / dims3, never
        %                     gets a halo - there's nothing there to borrow, and
        %                     single-pass fitting has the same one-sided edge)
        %   dims3           : total number of slices in the full volume, used to
        %                     clamp halo expansion at the true boundary
        %
        % Output
        %   seg : struct array with .owned, .fit, .local per segment (see
        %         find_optimal_segment_3D for field definitions). With h==0 this
        %         reduces to seg(k).fit == seg(k).owned == seg(k).local-shifted-
        %         to-1-based, i.e. legacy behaviour.

            NSegment = numel(ownedBoundaries);
            seg(NSegment) = struct('owned', [], 'fit', [], 'local', []);

            for ks = 1:NSegment
                owned = ownedBoundaries{ks};

                % Halo only on internal cut faces: segment 1 has no halo on its
                % low side (it IS the volume boundary), last segment has none on
                % its high side, for the same reason.
                loHalo = h * (ks > 1);
                hiHalo = h * (ks < NSegment);

                fitStart = max(1,     owned(1)   - loHalo);
                fitEnd   = min(dims3, owned(end) + hiHalo);
                fitRange = fitStart:fitEnd;

                seg(ks).owned = owned;
                seg(ks).fit   = fitRange;
                seg(ks).local = owned - fitRange(1) + 1;
            end

            % Warn when the halo eats a large fraction of the thinnest owned
            % segment - this is the combination that user-forced NSegment (high
            % NSegmentUser) plus nonzero overlap can produce: segments that
            % mostly refit their neighbours rather than their own data.
            if h > 0
                ownedThickness = cellfun(@numel, ownedBoundaries);
                thinnest       = min(ownedThickness);
                if h >= thinnest / 2
                    warning('GACELLE:haloVsOwnedThickness', ...
                        ['Halo width (h=%d) is large relative to the thinnest owned segment ' ...
                         '(%d slices). Segments will spend most of their compute refitting ' ...
                         'neighbouring (halo) slices rather than their own owned slices. ' ...
                         'Consider reducing fitting.segmentOverlap or fitting.NSegmentUser.'], ...
                        h, thinnest);
                end
            end
        end
        
        function [peakDelta_MiB,baseline_MiB,peak_MiB] = read_peak_from_log(logFile)
        % function peakDelta_MiB = read_peak_from_log(logFile)
        % get the peak memory usage from nvidia-smi log file

            try
                attempts = 0;
                while ~isfile(logFile) && attempts < 20
                    pause(0.05);
                    attempts = attempts + 1;
                end
        
                T    = readtable(logFile, 'FileType', 'text', 'Delimiter', ',', ...
                                 'VariableNamingRule', 'preserve');
                vals = T{:,1};
        
                if ~isnumeric(vals)
                    vals = str2double(erase(string(vals), ' MiB'));
                end
                vals = vals(~isnan(vals));
        
                if numel(vals) < 5
                    warning('GACELLE:memoryLog', ...
                        'Too few samples in log (%d) after reading %s.', numel(vals), logFile);
                    peak_MiB = max(vals);
                    return
                end
        
                % --- Discard startup noise ---
                % First 5% of samples may reflect nvidia-smi startup latency
                % and fit initialisation transients rather than steady pre-fit state
                nDiscard = max(2, round(0.05 * numel(vals)));
                vals     = vals(nDiscard+1:end);
        
                if numel(vals) < 3
                    warning('GACELLE:memoryLog', ...
                        'Too few samples after discarding startup portion (%d remaining).', numel(vals));
                    peak_MiB = max(vals);
                    return
                end
        
                % --- Baseline: median of next 10% after discarded portion ---
                % Median is robust to any residual transients in the early window
                nBaseline    = max(3, round(0.10 * numel(vals)));
                baseline_MiB = median(vals(1:nBaseline));
        
                % --- Peak: maximum of remaining samples after baseline window ---
                peak_MiB     = max(vals(nBaseline+1:end));
        
                % --- Delta: memory cost attributable to the fit ---
                peakDelta_MiB = peak_MiB - baseline_MiB;
        
                if peakDelta_MiB < 0
                    warning('GACELLE:memoryLog', ...
                        'Negative delta (baseline=%.0f, peak=%.0f MiB). Background activity may be interfering.', ...
                        baseline_MiB, peak_MiB);
                    peakDelta_MiB = 0;
                end
        
                fprintf('    [mem log] samples=%d, discarded=%d, baseline=%.0f MiB, peak=%.0f MiB, delta=%.0f MiB\n', ...
                    numel(vals)+nDiscard, nDiscard, baseline_MiB, peak_MiB, peakDelta_MiB);
        
            catch ME
                warning('GACELLE:memoryLog', 'Could not read log %s: %s', logFile, ME.message);
                g             = gpuDevice;
                peakDelta_MiB = (g.TotalMemory - g.FreeMemory) / 1024^2;
            end
        end
        
        function [memory, time, cmd_output] = run_and_profile(cmd, interval_ms)
        % =========================================================================
        % utils.run_and_profile  —  profile GPU memory and wall-clock time for any
        %                           GACELLE command run from the caller workspace.
        %
        % USAGE
        %   [memory, time, cmd_output] = utils.run_and_profile(cmd)
        %   [memory, time, cmd_output] = utils.run_and_profile(cmd, interval_ms)
        %
        % INPUT
        %   cmd         : (char) Command string exactly as you would type it at the
        %                 command line or in a script, e.g.:
        %                   'out = objGPU.estimate(y, mask, [], fitting);'
        %                 All variables referenced must exist in the caller
        %                 workspace. The function uses evalin('caller',...) so any
        %                 variable visible to the calling script is accessible.
        %                 NOTE: this utility is designed to be called directly from
        %                 a script or the command line, not from inside a function.
        %
        %   interval_ms : (optional, default 500) nvidia-smi polling interval in
        %                 milliseconds. Smaller values give finer time resolution
        %                 but produce larger log files for long runs.
        %
        % OUTPUT
        %   memory      : struct with fields
        %                   .total_MB       — 1-D array of raw nvidia-smi GPU memory
        %                                     readings (MiB), one per poll, all
        %                                     processes on the GPU
        %                   .matlab_MB      — 1-D array of best-estimate MATLAB-only
        %                                     memory (total minus other-process peak).
        %                                     Assumes other-process usage is roughly
        %                                     constant during the run; less reliable
        %                                     on busy shared HPC nodes
        %                   .timestamps_s   — 1-D array of elapsed seconds matching
        %                                     each reading (relative to cmd start)
        %                   .peak_total_MB  — scalar peak of total_MB
        %                   .peak_matlab_MB — scalar peak of matlab_MB
        %
        %   time        : struct with fields
        %                   .elapsed_s    — wall-clock elapsed time in seconds
        %
        %   cmd_output  : the variable assigned by cmd, if cmd assigns exactly one
        %                 output variable (detected by parsing the left-hand side).
        %                 Empty if cmd assigns nothing or assigns multiple variables.
        %
        % EXAMPLE
        %   fitting.iteration = 4000;
        %   [mem, t, out] = utils.run_and_profile( ...
        %       'out = objGPU.estimate(y, mask, [], fitting);', 200);
        %
        %   figure;
        %   plot(mem.timestamps_s, mem.total_MB, 'DisplayName', 'Total (all processes)');
        %   hold on;
        %   plot(mem.timestamps_s, mem.matlab_MB, 'DisplayName', 'MATLAB only (estimated)');
        %   xlabel('Time (s)'); ylabel('GPU memory used (MiB)');
        %   legend; title('GPU memory profile');
        %   fprintf('Peak GPU memory (total):  %.0f MiB\n', mem.peak_total_MB);
        %   fprintf('Peak GPU memory (MATLAB): %.0f MiB\n', mem.peak_matlab_MB);
        %   fprintf('Elapsed time:             %.1f s\n',   t.elapsed_s);
        % =========================================================================
            % --- defaults ---------------------------------------------------------
            if nargin < 2 || isempty(interval_ms)
                interval_ms = 500;
            end
        
            % --- parse LHS variable name from cmd --------------------------------
            % Matches single-variable assignment: out = ...
            % Multi-output patterns [a,b] = ... are not supported; cmd_output = [].
            lhs_token  = regexp(cmd, '^\s*(\w+)\s*=', 'tokens', 'once');
            has_lhs    = ~isempty(lhs_token);
            lhs_name   = '';
            if has_lhs
                lhs_name = lhs_token{1};
            end
        
            % --- get MATLAB PID for other-process subtraction --------------------
            matlabPID = feature('getpid');
        
            % --- prepare nvidia-smi log ------------------------------------------
            logFile = strcat(tempname, '_gacelle_profile.csv');
        
            cmd_smi = sprintf( ...
                'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms %d > %s &', ...
                round(interval_ms), logFile);
        
            system(cmd_smi);
        
            % Give nvidia-smi a moment to start and write its first line
            pause(0.05);
        
            % Retrieve nvidia-smi PID for clean shutdown
            [~, pidStr] = system('pgrep -n nvidia-smi');
            nvidiaPID   = strtrim(pidStr);
        
            % --- snapshot other-process memory before cmd ------------------------
            otherMem_before_MB = utils.get_other_process_memory(matlabPID);
        
            % --- run the command -------------------------------------------------
            t_start = tic;
        
            try
                evalin('caller', cmd);
            catch ME
                utils.run_and_profile_cleanup_(nvidiaPID);
                if isfile(logFile); delete(logFile); end
                rethrow(ME);
            end
        
            elapsed_s = toc(t_start);
        
            % --- snapshot other-process memory after cmd -------------------------
            otherMem_after_MB = utils.get_other_process_memory(matlabPID);
        
            % Conservative estimate: assume worst-case other-process usage during run
            otherMem_peak_MB = max(otherMem_before_MB, otherMem_after_MB);
        
            % --- shut down nvidia-smi --------------------------------------------
            utils.run_and_profile_cleanup_(nvidiaPID);
        
            % --- parse log and compute MATLAB-only series ------------------------
            memory = utils.run_and_profile_parse_log_(logFile, elapsed_s, otherMem_peak_MB);
        
            % --- package time ----------------------------------------------------
            time.elapsed_s = elapsed_s;
        
            % --- retrieve cmd output from caller workspace -----------------------
            cmd_output = [];
            if has_lhs
                try
                    cmd_output = evalin('caller', lhs_name);
                catch
                    % variable may not have been created (e.g. cmd errored quietly)
                end
            end
        
            % --- clean up log ----------------------------------------------------
            if isfile(logFile); delete(logFile); end
        
        end % run_and_profile

        function run_and_profile_cleanup_(nvidiaPID)
        % Kill the background nvidia-smi process and wait briefly for it to flush.
            if ~isempty(nvidiaPID)
                system(sprintf('kill %s 2>/dev/null', nvidiaPID));
            end
            pause(0.15);   % allow final writes to flush to disk
        end

        function memory = run_and_profile_parse_log_(logFile, elapsed_s, otherMem_peak_MB)
        % Read nvidia-smi CSV log and return the memory struct.
        %
        % Timestamps are assigned by uniform spacing across elapsed_s — a
        % reasonable approximation for diagnostic use at typical polling intervals.
        
            memory = struct( ...
                'total_MB',       [], ...
                'matlab_MB',      [], ...
                'timestamps_s',   [], ...
                'peak_total_MB',  NaN, ...
                'peak_matlab_MB', NaN);
        
            if ~isfile(logFile)
                warning('GACELLE:run_and_profile:noLog', ...
                    'nvidia-smi log not found. Is nvidia-smi available on this system?');
                return
            end
        
            try
                raw = readtable(logFile, ...
                    'ReadVariableNames', false, ...
                    'FileType',          'text', ...
                    'Delimiter',         '\n');
                vals = raw{:,1};
        
                if iscell(vals)
                    vals = cellfun(@str2double, vals);
                end
        
                vals(isnan(vals)) = [];
        
                if isempty(vals)
                    warning('GACELLE:run_and_profile:emptyLog', ...
                        'nvidia-smi log is empty. Polling may not have captured any samples.');
                    return
                end
        
                n          = numel(vals);
                timestamps = linspace(0, elapsed_s, n)';
                matlab_MB  = max(0, vals - otherMem_peak_MB);
        
                memory.total_MB       = vals(:);
                memory.matlab_MB      = matlab_MB(:);
                memory.timestamps_s   = timestamps;
                memory.peak_total_MB  = max(vals);
                memory.peak_matlab_MB = max(matlab_MB);
        
            catch ME
                warning('GACELLE:run_and_profile:parseFail', ...
                    'Failed to parse nvidia-smi log: %s', ME.message);
            end
        
        end % run_and_profile_parse_log_

        %%%%%%%%%%%%%%%%%%%
        % compute masked statistics
        function val = min_masked(img,mask)
            if size(mask,ndims(mask)) ~= size(img,ndims(img))
                mask = repmat(mask,[ones(1,ndims(img)-1) ndims(img)]);
            end
            val = min(img(mask>0));
        end

        function val = max_masked(img,mask)
            if size(mask,ndims(mask)) ~= size(img,ndims(img))
                mask = repmat(mask,[ones(1,ndims(img)-1) ndims(img)]);
            end
            val = max(img(mask>0));
        end

        function val = prctile_masked(img,mask,percentile)
            if size(mask,ndims(mask)) ~= size(img,ndims(img))
                mask = repmat(mask,[ones(1,ndims(img)-1) ndims(img)]);
            end
            val = prctile(img(mask>0),percentile);
        end

        function val = mean_masked(img,mask)
            if size(mask,ndims(mask)) ~= size(img,ndims(img))
                mask = repmat(mask,[ones(1,ndims(img)-1) size(img,ndims(img))]);
            end
            val = mean(img(mask>0));
        end

        function val = nnz(img)
            val = numel(img(img~=0));
        end

        % % This function create a full out structure variable if the data is divided into multiple segments
        % function out = restore_segment_structure(out,out_tmp,slice,ksegment)
        % % Input
        % % ---------
        % % out       : askadam out structure final output 
        % % out_tmp   : temporary out structure of each segment
        % % slice     : slices where the segment belongs to
        % % ksegment  : current segment number
        % % 
        % 
        %     % reformat out structure
        %     fn1 = fieldnames(out_tmp);
        %     for kfn1 = 1:numel(fn1)
        %         fn2 = fieldnames(out_tmp.(fn1{kfn1}));
        %         for kfn2 = 1:numel(fn2)
        %             if isscalar(out_tmp.(fn1{kfn1}).(fn2{kfn2})) % scalar value
        %                 out.(fn1{kfn1}).(fn2{kfn2})(ksegment) = out_tmp.(fn1{kfn1}).(fn2{kfn2});
        %             else
        %                 % image result
        %                 try
        %                     if ksegment == 1
        %                         out.(fn1{kfn1}).(fn2{kfn2}) = out_tmp.(fn1{kfn1}).(fn2{kfn2});
        %                     else
        %                         out.(fn1{kfn1}).(fn2{kfn2})(:,:,slice,:,:) = out_tmp.(fn1{kfn1}).(fn2{kfn2});
        %                     end
        %                 catch
        %                     if ksegment == 1
        %                         out.(fn1{kfn1}).(fn2{kfn2}) = out_tmp.(fn1{kfn1}).(fn2{kfn2});
        %                     else
        %                         out.(fn1{kfn1}).(fn2{kfn2}) = cat(1,out.(fn1{kfn1}).(fn2{kfn2}) ,out_tmp.(fn1{kfn1}).(fn2{kfn2}));
        %                     end
        %                 end
        %             end
        % 
        %         end
        %     end
        % end

        function out = restore_segment_structure(out, out_tmp, slice, ksegment)
        % Restore segmented fitting output into a single output structure.
        %
        % Handles fields of any shape:
        %   - Scalar         : collected into a vector across segments
        %   - 3D image       : inserted at correct slice indices along dim 3
        %   - ND image       : inserted at correct slice indices along dim 3
        %   - 2D matrix      : concatenated along dim 2 (voxel dimension in askadam)
        %   - 1D vector      : concatenated along dim 1
        %   - Char/string    : kept from first segment only
        %   - Nested struct  : recursed into (handles out.final, out.min, etc.)
        
            fn1 = fieldnames(out_tmp);
            for kfn1 = 1:numel(fn1)
                field1 = fn1{kfn1};
                val    = out_tmp.(field1);
        
                if isstruct(val)
                    % Recurse one level (handles out.final.X, out.min.X, etc.)
                    fn2 = fieldnames(val);
                    for kfn2 = 1:numel(fn2)
                        field2 = fn2{kfn2};
                        out.(field1).(field2) = utils.restore_field( ...
                            utils.get_field_safe(out, field1, field2), ...
                            val.(field2), slice, ksegment);
                    end
                else
                    % Top-level non-struct field (e.g. out.mask)
                    out.(field1) = utils.restore_field( ...
                        utils.get_field_safe(out, field1), ...
                        val, slice, ksegment);
                end
            end
        end

        function existing = get_field_safe(out, varargin)
        % Safely retrieve a (possibly absent) field, returning [] if missing.
            try
                existing = out;
                for k = 1:numel(varargin)
                    existing = existing.(varargin{k});
                end
            catch
                existing = [];
            end
        end
        
        function merged = restore_field(existing, new_val, slice, ksegment)
        % Merge new_val from one segment into the existing accumulated value.
        
            % Non-numeric types: keep from first segment, ignore subsequent
            if ischar(new_val) || isstring(new_val) || islogical(new_val) && isscalar(new_val)
                if ksegment == 1
                    merged = new_val;
                else
                    merged = existing;
                end
                return
            end
        
            % Empty: pass through
            if isempty(new_val)
                merged = new_val;
                return
            end
        
            sz  = size(new_val);
            nd  = ndims(new_val);
        
            % --- Scalar numeric ---
            if isscalar(new_val)
                % Accumulate one value per segment into a growing vector
                if ksegment == 1
                    merged = new_val;
                else
                    merged = [existing, new_val];
                end
                return
            end
        
            % --- Spatially-indexed array: 3rd dim matches slice count ---
            % This covers 3D, 4D, 5D parameter maps
            if nd >= 3 && sz(3) == numel(slice)
                if ksegment == 1
                    % Pre-allocate full output on first segment using total slice info
                    % We don't know total slices yet so just store; will expand later
                    merged = new_val;
                else
                    % Insert at correct slice positions along dim 3
                    % Works for any number of trailing dimensions
                    idx                  = repmat({':'}, 1, nd);
                    idx{3}               = slice;
                    merged               = existing;
                    merged(idx{:})       = new_val;
                end
                return
            end
        
            % --- 2D matrix: rows=measurements, cols=voxels (askadam residual format) ---
            if nd == 2 && sz(2) > 1
                if ksegment == 1
                    merged = new_val;
                else
                    merged = cat(2, existing, new_val);  % concatenate along voxel dim
                end
                return
            end
        
            % --- 1D vector ---
            if nd == 2 && sz(2) == 1
                if ksegment == 1
                    merged = new_val;
                else
                    merged = cat(1, existing, new_val);
                end
                return
            end
        
            % --- Fallback: store from first segment, warn on subsequent ---
            if ksegment == 1
                merged = new_val;
            else
                warning('GACELLE:restoreSegment', ...
                    'Cannot determine how to merge field of size [%s] along segments. Keeping first segment value.', ...
                    num2str(sz));
                merged = existing;
            end
        end

        function out = crop_segment_output(out_tmp, seg)
        % Crop a segment's FIT (haloed) output down to its OWNED sub-range along
        % the slice axis, before restore_segment_structure writes it back at
        % seg.owned. Mirrors restore_segment_structure's struct-walk shape, but
        % reads (crops) instead of writes (inserts).
        %
        % Input
        %   out_tmp : output structure from this.fit() on the haloed segment,
        %             i.e. spatially indexed 1:numel(seg.fit) along the slice axis
        %   seg     : single segment's geometry struct, with .owned, .fit, .local
        %             as returned by find_optimal_segment_3D / expand_segments_with_halo
        %
        % When seg.fit and seg.owned are identical (h==0, or this is a 1-segment
        % run), crop_field below is a no-op (localRange == 1:numel(fitRange)), so
        % this function has zero effect on legacy (no-halo) runs.

            fn1 = fieldnames(out_tmp);
            for kfn1 = 1:numel(fn1)
                field1 = fn1{kfn1};
                val    = out_tmp.(field1);

                if isstruct(val)
                    fn2 = fieldnames(val);
                    for kfn2 = 1:numel(fn2)
                        field2 = fn2{kfn2};
                        out.(field1).(field2) = utils.crop_field(val.(field2), seg);
                    end
                else
                    out.(field1) = utils.crop_field(val, seg);
                end
            end
        end

        function cropped = crop_field(val, seg)
        % Crop val along its slice axis from seg.fit-indexing down to seg.local
        % (the owned sub-range within the fitted segment), using the same shape
        % dispatch order as restore_field so the two functions interpret any
        % given field's shape identically and cannot silently diverge.
        %
        % Fields with no slice axis under this dispatch (scalars accumulated
        % per-segment, and the 2D [Nmeasurement, Nvoxel] voxel-flattened askadam
        % residual format) are passed through unchanged, with no crop and no
        % warning - same treatment restore_field already gives this shape
        % (concatenation along the voxel dimension, not slice-axis insertion),
        % since it is voxel-flattened by design and never carries a contiguous
        % slice axis to begin with.

            localRange  = seg.local;
            fitRangeLen = numel(seg.fit);

            % Non-numeric / scalar-logical: nothing to crop, identical to restore_field's
            % handling - these are kept-from-first-segment values, not per-slice data.
            if ischar(val) || isstring(val) || (islogical(val) && isscalar(val))
                cropped = val;
                return
            end

            if isempty(val)
                cropped = val;
                return
            end

            sz = size(val);
            nd = ndims(val);

            % --- Scalar numeric: no slice axis to crop, pass through ---
            if isscalar(val)
                cropped = val;
                return
            end

            % --- Spatially-indexed array: 3rd dim matches the FIT (haloed) extent ---
            if nd >= 3 && sz(3) == fitRangeLen
                idx        = repmat({':'}, 1, nd);
                idx{3}     = localRange;
                cropped    = val(idx{:});
                return
            end

            % --- 2D voxel-flattened (askadam residual format), e.g. [Nmeasurement,
            % Nvoxel]: no slice axis by design (kept unmasked/flattened specifically
            % to avoid the memory cost of a 4D/5D spatial representation). Pass
            % through unchanged - restore_segment_structure already concatenates
            % this shape correctly along the voxel dimension regardless of crop.
            if nd == 2 && sz(2) > 1
                cropped = val;
                return
            end

            % --- 1D vector: same - no slice axis, pass through ---
            if nd == 2 && sz(2) == 1
                cropped = val;
                return
            end

            % --- Fallback: unrecognised shape, pass through ---
            cropped = val;
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
    
         % make sure all network parameters stay between 0 and 1
        function parameters = set_boundary(parameters,ub,lb)

            field = fieldnames(parameters);
            for k = 1:numel(field)
                parameters.(field{k})   = max(parameters.(field{k}),lb(k)); % Lower bound     
                parameters.(field{k})   = min(parameters.(field{k}),ub(k)); % upper bound

            end

        end
    
    end

end