classdef gpuIVIM < handle
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 28 September 2026

    properties (GetAccess = public, SetAccess = protected)
    % ===== MODEL PARAMETER CONTRACT =====
    % S0        : signal at b = 0, relative to the mean signal at the lowest b-value (see estimate)
    % f         : perfusion (pseudo-diffusion) fraction
    % D         : tissue diffusion coefficient [um2/ms]
    % Dstar     : pseudo-diffusion coefficient [um2/ms]
    % noise     : noise
    %
    % modelParams{k} <-> ub(k) <-> lb(k) <-> startPoint(k) <-> step(k)
    % These five arrays MUST stay the same length and index-aligned.
    % Mutate only as a set, via updateProperty() - never assign into a
    % single element from outside the class, or these will desync.
    %
    % 'noise' is solver-conditional (mcmc only) and is kept LAST so that
    % updateProperty() can strip it by name without hardcoding an index.
    % Bounds as in BayesIVIM (Spinner et al. 2021), suited to brain IVIM (Dstar of a few um2/ms);
    % Dstar > D is not enforced, so D and Dstar can swap in poorly determined voxels.
        modelParams     = { 'S0';  'f';    'D'; 'Dstar'; 'noise'};
        ub              = [    2;    1;    2.5;      50;     0.1];
        lb              = [    0;    0;      0;       0;   0.001];
        startPoint      = [    1;  0.1;      1;      20;    0.01];
        step            = [ 0.01; 0.02;   0.02;       1;   0.005];

    end

    properties
    % ===== USER-TUNABLE OPTIONS =====
    % Freely settable by users before fitting; no coupling between these.
        thres_bkg   = 0.01;     % voxels whose lowest-b signal is below thres_bkg * (99th percentile) are excluded

        epsilon     = utils.epsilon;

    end

    properties (GetAccess = public, SetAccess = protected)
    % ===== ACQUISITION PARAMETERS =====
    % Set once in the constructor from user-provided acquisition info.
    % Read-only after construction.
        b;
    end

    methods

        % constructuor
        function this = gpuIVIM(b)
        % Intravoxel incoherent motion (IVIM) bi-exponential model
        %   S(b) = S0 * [ f*exp(-b*Dstar) + (1-f)*exp(-b*D) ]
        % Le Bihan D, Breton E, Lallemand D, Aubin ML, Vignaud J, Laval-Jeantet M. Separation of
        % diffusion and perfusion in intravoxel incoherent motion MR imaging. Radiology 1988;168:497-505.
        % DOI: 10.1148/radiology.168.2.3393671
        %
        % obj = gpuIVIM(b)
        %       output:
        %           - obj: object of a fitting class
        %
        %       input:
        %           - b: b-value of each volume of the data, in the same order [ms/um2]
        %                (1000 s/mm2 = 1 ms/um2)
        %
        %  Authors:
        %  Kwok-Shing Chan (kchan2@mgh.harvard.edu)
        %
            b = double(b(:));
            if any(b < 0) || ~all(isfinite(b))
                error('gpuIVIM:bval', 'b-values must be finite and non-negative [ms/um2].');
            end
            if max(b) > 20
                error('gpuIVIM:bval', ['The largest b-value is %g: b must be in ms/um2 (1000 s/mm2 = 1 ms/um2). ' ...
                                       'Divide b in s/mm2 by 1000.'], max(b));
            end
            if numel(unique(b)) < 3
                error('gpuIVIM:bval', 'IVIM needs at least 3 distinct b-values.');
            end
            this.b = single(b);

        end

        % update properties according to the solver
        function this = updateProperty(this, fitting)

            % property change in related to solver
            if ~strcmpi(fitting.solver,'mcmc')
                idx = find(ismember(this.modelParams,'noise'));
                this.modelParams(idx)       = [];
                this.lb(idx)                = [];
                this.ub(idx)                = [];
                this.startPoint(idx)        = [];
                this.step(idx)              = [];
            end

        end

        % display some info about the input data and model parameters
        function display_data_model_info(this)

            disp('================================================');
            disp('Intravoxel incoherent motion (IVIM) bi-exponential model');
            disp('================================================');

            disp('----------------')
            disp('Data Information');
            disp('----------------')
            fprintf('b-values (ms/um2)              : [%s] \n',num2str(this.b.',' %.3f'));

        end

        %% higher-level data fitting functions

        % This is a wrapper of the 'fit' function.
        % The main purpose of this function is to handle memory issue and ensure the input data is correct for 'fit'
        function  [out] = estimate(this, data, mask, extraData, fitting, pars0)
        % Input data are expected in multi-dimensional image
        %
        % Input
        % -----------
        % data      : 4D DWI, [x,y,z,dwi], one volume per entry of b (constructor), same order
        %             (direction-averaged or trace-weighted images)
        % mask      : 3D signal mask, [x,y,z]
        % extradata : not used, kept for the common interface ([] is fine)
        % fitting   : fitting algorithm parameters
        %   .solver         : 'askadam' (default) | 'mcmc'
        %   .start          : starting points, 'prior' (default, segmented fit) | 'default' | 1xM array
        %   .bThreshold     : b-value [ms/um2] separating the diffusion (b >= bThreshold) and
        %                     perfusion regimes in the segmented start, default 0.2
        %   (see askadam/mcmc for the solver options; fitting.mcmcClass = 'mcmc_bayes' for the
        %    experimental Bayesian priors)
        % pars0     : structure of starting points, one field per model parameter (optional)
        %
        % Output
        % -----------
        % out       : output structure contains all parameter estimation results
        %   .signalScale : 3D map, mean signal at the lowest b-value used to normalise the data;
        %                  S0 in data units = out.<metric>.S0 .* out.signalScale
        %   .mask        : final signal mask
        %

            % display basic info
            this.display_data_model_info;

            % if no extraData input at all (not even empty) then assume none
            if nargin < 4; extraData    = []; end
            if nargin < 5; fitting      = struct(); end

            % get all fitting algorithm parameters
            fitting = this.check_set_default(fitting);
            [extraData, noiseMap] = mcmc_bayes.take_noise_map(extraData);    % extraData.noiseSigma (mcmc_bayes noise map), not seen by FWD

            %%%%%%%%%%%%%%%% Step 1: Validate all input data %%%%%%%%%%%%%%%%
            % if no pars input at all (not even empty) then use prior
            if nargin < 6; pars0        = []; end

            if size(data,4) ~= numel(this.b)
                error('gpuIVIM:dataSize', 'The data have %d volumes but the object has %d b-values.', size(data,4), numel(this.b));
            end

            % convert datatype to single or logical, normalise by the lowest-b signal
            data    = single(data);
            mask    = mask >0;
            [data, mask, signalScale] = this.prepare_data(data, mask);
            if ~isempty(pars0); for km = 1:numel(this.modelParams); pars0.(this.modelParams{km}) = single(pars0.(this.modelParams{km})); end; end

            %%%%%%%%%%%%%%%% End Step 1 %%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%% Step 2: Memory management %%%%%%%%%%%%%%%%

            % --- [Experimental] estimate memory usage using a small batch of data size ---
            % this method tends to be more conservative than the actual memory ussage
            fitting = mcmc_bayes.noise_map_to_fitting(fitting, noiseMap, mask, signalScale);   % -> fitting.ricianSigma, fitted-data units
            [seg,NSegment] = utils.find_optimal_segment_3D(this, data, mask, fitting, extraData);
            % priors coupling voxels (free hierarchical, MRF) need the whole volume in one call
            if strcmpi(fitting.solver,'mcmc') && strcmpi(fitting.mcmcClass,'mcmc_bayes') && NSegment > 1 && mcmc_bayes.needs_single_segment(fitting)
                error('gpuIVIM:singleSegment', ...
                    ['The mcmc_bayes prior couples voxels (free hierarchical or MRF) and needs the whole volume in one ' ...
                     'GPU call, but the data were divided into %d segments. Reduce the volume or use fixed hyperparameters.'], NSegment);
            end

            % parameter estimation
            out = [];
            for kseg = 1:NSegment

                if NSegment > 1
                    fprintf('Running #Segment = %d/%d \n',kseg,NSegment);
                    disp   ('------------------------')
                end

                % divide the data; fitRange includes halo slices (if any), ownedRange
                % is what this segment is responsible for writing back
                fitRange                        = seg(kseg).fit;
                ownedRange                      = seg(kseg).owned;
                [dataSeg, maskSeg,extraDataSeg,pars0Seg]    = this.slice_segment(data, mask, fitRange, extraData, pars0);

                % run fitting
                [outSeg] = this.fit(dataSeg,maskSeg,mcmc_bayes.slice_noise_map(fitting, fitRange, size(mask,1:3)),extraDataSeg,pars0Seg);

                % discard halo slices from this segment's output before restoring
                outSeg = utils.crop_segment_output(outSeg, seg(kseg));

                % restore 'out' structure from segment
                out = utils.restore_segment_structure(out,outSeg,ownedRange,kseg);

            end
            out.mask        = mask;
            out.signalScale = signalScale;
            %%%%%%%%%%%%%%%% End Step 2 %%%%%%%%%%%%%%%%

            % save the estimation results if the output filename is provided
            switch fitting.solver
                case 'askadam'
                    askadam.save_askadam_output(fitting.outputFilename,out)
                case 'mcmc'
                    mcmc.save_mcmc_output(fitting.outputFilename,out)
            end

        end

        % Data fitting function
        function [out] = fit(this,data,mask,fitting, extraData, pars0)
        %
        % Input
        % -----------
        % data      : 4D DWI normalised by the lowest-b signal, [x,y,z,dwi]
        % mask      : 3D signal mask, [x,y,z]
        % fitting   : fitting algorithm parameters (see estimate, askadam and mcmc)
        % pars0     : structure variable of starting points of fitting (optional)
        %
        % Output
        % -----------
        % out       : output structure of askadam or mcmc
        %

            % check GPU
            gpool = gpuDevice;

            % check image size
            dims = size(data,1:3);

            %%%%%%%%%%%%%%%%%%%% Step 1. Validate and parse input %%%%%%%%%%%%%%%%%%%%
            if nargin < 3 || isempty(mask); mask = ones(dims,'logical'); end % if no mask input then fit everthing
            if nargin < 4; fitting = struct(); end
            if nargin < 5; extraData = []; end %#ok<NASGU>
            % set initial tarting points
            if nargin < 6; pars0 = []; % no initial starting points
            else
                if ~isempty(pars0); for km = 1:numel(this.modelParams); pars0.(this.modelParams{km}) = single(pars0.(this.modelParams{km})); end; end
            end

            % get all fitting algorithm parameters
            fitting                 = this.check_set_default(fitting);
            % determine fitting parameters
            this                    = this.updateProperty(fitting);
            fitting.modelParams     = this.modelParams;
            % set fitting boundary if no input from user
            if isempty( fitting.ub); fitting.ub = this.ub(1:numel(fitting.modelParams)); end
            if isempty( fitting.lb); fitting.lb = this.lb(1:numel(fitting.modelParams)); end

            %%%%%%%%%%%%%%%%%%%% End 1 %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% 2. Setting up all necessary data, run askadam and get all output %%%%%%%%%%%%%%%%%%%%
            % 2.1 setup fitting weights (uniform)
            w = ones(size(data),'single');

            % 2.2 estimate prior if needed
            if isempty(pars0);  pars0 = this.determine_x0(data,mask,fitting); end

            % 2.3 optimisation main
            switch fitting.solver
                case 'askadam'
                    out         = askadam().optimisation(data, mask, w, pars0, fitting, @this.FWD, fitting.solver, fitting);
                case 'mcmc'
                    fitting.xStepSize = this.step;

                    out         = feval(fitting.mcmcClass).optimisation(data, mask, w, pars0, fitting, @this.FWD, fitting.solver, fitting);
            end
            %%%%%%%%%%%%%%%%%%%% End 2 %%%%%%%%%%%%%%%%%%%%

            disp('The estimation is completed.');

            % clear GPU
            reset(gpool)

        end

        %% Data preparation

        % normalise by the mean signal at the lowest b-value; exclude NaN/Inf and background voxels
        function [data, mask, signalScale] = prepare_data(this, data, mask)

            [data, mask]    = utils.remove_img_naninf(data, mask);

            isLowest        = this.b == min(this.b);
            signalScale     = mean(data(:,:,:,isLowest), 4);
            v               = sort(signalScale(mask));      % 99th percentile without the Statistics toolbox
            if isempty(v); v = 0; end
            thres           = this.thres_bkg * v(max(1, ceil(0.99*numel(v))));
            mask_bkg        = signalScale <= max(thres, 0);
            if any(mask(:) & mask_bkg(:))
                fprintf('Signal mask updated: %d background voxels excluded (lowest-b signal <= %.3g).\n', nnz(mask & mask_bkg), thres);
                disp('Please use the updated mask (out.mask) in subsequent analysis.');
            end
            mask            = mask & ~mask_bkg;
            signalScale     = signalScale .* mask;
            data            = data ./ max(signalScale, eps('single'));
            data(~repmat(mask, [1 1 1 size(data,4)])) = 0;

        end

        %%%%% Prior estimation related functions %%%%%

        % determine how the starting points will be set up
        function x0 = determine_x0(this,y,mask,fitting)

            disp('---------------');
            disp('Starting points');
            disp('---------------');

            dims = size(mask,1:3);

            if ischar(fitting.start)
                switch lower(fitting.start)
                    case 'prior'
                        % segmented fit
                        x0 = this.estimate_prior(y,mask,fitting);

                    case 'default'
                        % use fixed points
                        fprintf('Using default starting points for all voxels at [%s]: [%s]\n', cell2str(this.modelParams),replace(num2str(this.startPoint(:).',' %.2f'),' ',','));
                        x0 = utils.initialise_x0(dims,this.modelParams,this.startPoint);

                    otherwise
                        error('gpuIVIM:start', 'fitting.start must be ''prior'', ''default'' or a numeric array (got ''%s'').', fitting.start);
                end
            else
                % user defined starting point
                x0 = fitting.start(:);
                fprintf('Using user-defined starting points for all voxels at [%s]: [%s]\n',cell2str(this.modelParams),replace(num2str(x0(:).',' %.2f'),' ',','));
                x0 = utils.initialise_x0(dims,this.modelParams,x0);

            end

            % make sure the input is bounded
            x0 = mcmc.set_boundary(x0,fitting.ub,fitting.lb);

            fprintf('Estimation lower bound [%s]: [%s]\n',      cell2str(this.modelParams),replace(num2str(fitting.lb(:).',' %.2f'),' ',','));
            fprintf('Estimation upper bound [%s]: [%s]\n',      cell2str(this.modelParams),replace(num2str(fitting.ub(:).',' %.2f'),'  ',','));
            disp('---------------');
        end

        % segmented fit for starting points
        function pars0 = estimate_prior(this, y, mask, fitting)
        % 1. D and the intercept A from a log-linear fit on b >= bThreshold (perfusion negligible)
        % 2. S0 = mean signal at the lowest b-value, f = 1 - A/S0
        % 3. Dstar from a log-linear fit of the residual S - A*exp(-b*D) on b < bThreshold
        % Voxels where a step is not possible (too few b-values, non-positive residual) keep the
        % default starting point of that parameter.

            disp('Estimate starting points using a segmented fit ...')

            dims    = size(mask,1:3);
            Nv      = prod(dims);
            b       = double(this.b(:));
            S       = double(reshape(y, Nv, numel(b)).');       % [Nb, Nv]

            pars    = repmat(this.startPoint(:), 1, Nv);         % [Nparam, Nv]
            iS0 = 1; iF = 2; iD = 3; iDs = 4;

            % 1. diffusion regime
            hi = b >= fitting.bThreshold;
            if numel(unique(b(hi))) >= 2
                [slope, icpt] = gpuIVIM.loglinear(b(hi), S(hi,:));
                D       = -slope;
                A       = exp(icpt);
                ok      = isfinite(D) & D > 0;
                pars(iD, ok) = D(ok);
            else
                warning('gpuIVIM:segmented', 'Fewer than 2 distinct b-values >= bThreshold (%g): D keeps its default start.', fitting.bThreshold);
                A       = nan(1, Nv);
            end

            % 2. S0 and f
            S0              = mean(S(b == min(b), :), 1);
            pars(iS0, :)    = S0;
            f               = 1 - A ./ S0;
            ok              = isfinite(f);
            pars(iF, ok)    = f(ok);

            % 3. perfusion regime: residual after removing the diffusion component
            lo = b < fitting.bThreshold;
            if numel(unique(b(lo))) >= 2
                R       = S(lo,:) - A .* exp(-b(lo) * pars(iD,:));
                R(R <= 0) = NaN;
                [slope] = gpuIVIM.loglinear(b(lo), R);
                ok      = isfinite(slope) & -slope > 0;
                pars(iDs, ok) = -slope(ok);
            end

            % bound and reshape
            lbP = fitting.lb(1:size(pars,1)); ubP = fitting.ub(1:size(pars,1));      % the fitting bounds (user or class default)
            pars = min(max(pars, lbP(:)), ubP(:));
            for km = 1:numel(this.modelParams)
                if strcmp(this.modelParams{km}, 'noise')
                    pars0.noise = single(ones(dims) * this.startPoint(km));
                else
                    pars0.(this.modelParams{km}) = single(reshape(pars(km,:), dims) .* mask);
                end
            end

        end

        % segment data based on slice
        function [dataSeg, maskSeg, extraDataSeg, pars0Seg] = slice_segment(this, data, mask, slice, extraData, pars0)

            dataSeg     = data(:,:,slice,:,:,:,:,:,:);
            maskSeg     = mask(:,:,slice);
            extraDataSeg= [];

            if ~isempty(pars0)
                for km = 1:numel(this.modelParams)
                    if isfield(pars0, this.modelParams{km})
                        pars0Seg.(this.modelParams{km}) = pars0.(this.modelParams{km})(:,:,slice);
                    end
                end
            else
                pars0Seg = [];
            end

            if ~isempty(extraData)
                fields      = fieldnames(extraData);
                for kfield = 1:numel(fields)
                    if ~ismatrix(extraData.(fields{kfield}))
                        extraDataSeg.(fields{kfield}) = extraData.(fields{kfield})(:,:,slice,:,:,:,:,:,:,:,:);
                    end
                end
            end

        end

        %%  Signal related functions

        % Forward model
        function s = FWD(this, pars, solver, fitting) %#ok<INUSD>
        % pars fields are [1,Nv] (askadam, mcmc 'MH') or [1,Nv,Nwalker] (mcmc 'ensemble');
        % s is [Nb,Nv] or [Nb,Nv,Nwalker] accordingly. Without S0 (e.g. mcmc_bayes with
        % S0Param = 'S0') the amplitude is 1.

            b = this.b;                 % [Nb,1]
            if isfield(pars,'S0'); S0 = pars.S0; else; S0 = 1; end

            s = S0 .* gpuIVIM.ivim_signal(pars.f, pars.D, pars.Dstar, b);

        end

    end

    methods(Static)

        %% Utility

        function s = ivim_signal(f, D, Dstar, b)
            s = f .* exp(-b .* Dstar) + (1 - f) .* exp(-b .* D);
        end

        % least-squares line log(S) = icpt + slope*b per column, NaN entries of S ignored
        function [slope, icpt] = loglinear(b, S)
            L       = log(S);
            valid   = isfinite(L);
            n       = sum(valid, 1);
            B       = repmat(b(:), 1, size(S,2));
            B(~valid) = 0; L(~valid) = 0;
            sb      = sum(B,1);  sl = sum(L,1);
            sbb     = sum(B.^2,1); sbl = sum(B.*L,1);
            den     = n.*sbb - sb.^2;
            slope   = (n.*sbl - sb.*sl) ./ den;
            icpt    = (sl - slope.*sb) ./ n;
            bad     = n < 2 | den <= 0;
            slope(bad) = NaN; icpt(bad) = NaN;
        end

        % check and set default fitting algorithm parameters
        function fitting2 = check_set_default(fitting)
            % get basic fitting setting check
            if ~isfield(fitting,'solver');      fitting.solver = 'askadam';        end
            if strcmpi(fitting.solver,'askadam')
                fitting2 = askadam.check_set_default_basic(fitting);

                if ~isfield(fitting,'regmap');      fitting2.regmap = {'f'};            end

                if ~iscell(fitting2.regmap)
                    fitting2.regmap = cellstr(fitting2.regmap);
                end

            else
                fitting2                = mcmc.check_set_default_basic(fitting);
                % sampler class: 'mcmc' (default) or the experimental 'mcmc_bayes'
                if ~isfield(fitting,'mcmcClass');   fitting2.mcmcClass = 'mcmc';    end
                fitting2.lossFunction   = [];
            end

            % get customised fitting setting check
            if ~isfield(fitting,'start');       fitting2.start      = 'prior';      end
            if ~isfield(fitting,'bThreshold');  fitting2.bThreshold = 0.2;          end

        end

    end

end
