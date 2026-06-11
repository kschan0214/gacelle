classdef DWIutility

    properties (GetAccess = public, SetAccess = protected)
        % b;
        % Delta;
    end

    methods

        function [dwi_out, bval_out, ldelta_out, BDELTA_out, te_out] = compute_rotationally_invariant_signal(this, dwi, bval, bvec, ldelta, BDELTA, te, lmax)
        % COMPUTE_ROTATIONALLY_INVARIANT_SIGNAL
        %   Wrapper around get_Sl_all_no_normalise that handles b=0 normalisation,
        %   removes higher-order Sl from b=0 shells, and returns shell descriptors
        %   aligned with the 4th dimension of dwi_out.
        %
        % Behaviour depends on number of unique TEs:
        %   NTE = 1 : normalise by b=0 Sl0, discard b=0 from output
        %             dwi_out contains normalised non-b0 shells only
        %   NTE > 1 : no normalisation, keep b=0 in output (Sl0 only, l>0 removed)
        %             dwi_out contains all shells
        %
        % Output 4th dimension layout:
        %   [Sl0_sh1, Sl0_sh2, ..., Sl0_shN, Sl2_sh1, Sl2_sh2, ..., Sl2_shN]
        %   For NTE>1, b=0 shells contribute only to Sl0 block (one entry each)
        %   and are absent from Sl2 and higher blocks.
        %
        % bval_out/ldelta_out/BDELTA_out/te_out reflect the 4th dimension exactly.
        %
        % Input
        % -----
        %   dwi     : 4D DWI data [x, y, z, volumes]
        %   bval    : 1D b-value vector, one per volume [ms/um^2]
        %   bvec    : 2D gradient directions [3, volumes]
        %   ldelta  : 1D gradient pulse duration, one per volume [ms]
        %             Pass [] or scalar to broadcast
        %   BDELTA  : 1D diffusion time, one per volume [ms]
        %             Pass [] or scalar to broadcast
        %   te      : 1D echo time, one per volume [ms]
        %             Pass [] or scalar to broadcast
        %   lmax    : maximum spherical harmonic order (0 or 2), default = 0
        %
        % Output
        % ------
        %   dwi_out    : 4D rotationally invariant signal (see layout above)
        %   bval_out   : b-values aligned with 4th dim of dwi_out
        %   ldelta_out : little delta aligned with 4th dim of dwi_out
        %   BDELTA_out : big delta aligned with 4th dim of dwi_out
        %   te_out     : echo time aligned with 4th dim of dwi_out
        
            if nargin < 8; lmax = 0; end
        
            % --- Broadcast scalar/empty acquisition parameters to per-volume vectors ---
            if isempty(ldelta);  ldelta = ones(size(bval));          end
            if isscalar(ldelta); ldelta = ones(size(bval)) * ldelta; end
            if isempty(BDELTA);  BDELTA = ones(size(bval));          end
            if isscalar(BDELTA); BDELTA = ones(size(bval)) * BDELTA; end
            if isempty(te);      te     = zeros(size(bval));         end
            if isscalar(te);     te     = ones(size(bval))  * te;    end
        
            Nsl       = lmax/2 + 1;
            te_unique = unique(te);
            NTE       = numel(te_unique);
        
            fprintf('Computing rotationally invariant signal...');
        
            % --- Step 1: compute raw Sl signal ---
            % layout: [Sl0_sh1,...,Sl0_shN, Sl2_sh1,...,Sl2_shN]
            % shell order matches unique_shell_keepb0 by construction
            [dwi_raw, bval_loop] = this.get_Sl_all_no_normalise( ...
                dwi, bval, bvec, ldelta, BDELTA, te, lmax);
        
            Nshells = numel(bval_loop);   % total shells including b=0
            isb0    = bval_loop == 0;
        
            % --- Step 2: reconstruct shell descriptors in same loop order ---
            ldelta_loop = [];
            BDELTA_loop = [];
            te_loop     = [];
        
            for kt = 1:NTE
                idx_te = find(te == te_unique(kt));
        
                % b=0: smallest ldelta then smallest BDELTA within this TE
                if any(bval(idx_te) == 0)
                    ld_first    = min(ldelta(idx_te));
                    idx_ld1     = intersect(find(ldelta == ld_first), idx_te);
                    bd_first    = min(BDELTA(idx_ld1));
                    ldelta_loop = cat(1, ldelta_loop, ld_first);
                    BDELTA_loop = cat(1, BDELTA_loop, bd_first);
                    te_loop     = cat(1, te_loop,     te_unique(kt));
                end
        
                % non-b0 shells
                ldelta_unique = unique(ldelta(idx_te));
                for klde = 1:numel(ldelta_unique)
                    idx_ldel      = intersect(find(ldelta == ldelta_unique(klde)), idx_te);
                    BDELTA_unique = unique(BDELTA(idx_ldel));
                    for kBDE = 1:numel(BDELTA_unique)
                        idx_group = intersect(find(BDELTA == BDELTA_unique(kBDE)), idx_ldel);
                        b_unique  = unique(bval(idx_group));
                        b_unique  = b_unique(b_unique > 0);
                        if isempty(b_unique); continue; end
                        Nshells_g   = numel(b_unique);
                        ldelta_loop = cat(1, ldelta_loop, ones(Nshells_g,1) * ldelta_unique(klde));
                        BDELTA_loop = cat(1, BDELTA_loop, ones(Nshells_g,1) * BDELTA_unique(kBDE));
                        te_loop     = cat(1, te_loop,     ones(Nshells_g,1) * te_unique(kt));
                    end
                end
            end
        
            % --- Step 3: build output depending on NTE ---
            if NTE == 1
                % Normalise by b=0 Sl0 then discard b=0
                % b=0 is at index 1 in Sl0 block (first shell)
                dwi_b0 = dwi_raw(:,:,:, 1);   % Sl0 of b=0
        
                % Non-b0 indices in each Sl block
                % Sl0 block: indices 1:Nshells, non-b0 are 2:Nshells (b=0 is index 1)
                % Sl2 block: indices Nshells+1:2*Nshells, non-b0 are Nshells+2:2*Nshells
                % General: for block kl (0-based), non-b0 indices are:
                %          kl*Nshells + find(~isb0)
                nonb0_pos  = find(~isb0);   % positions of non-b0 shells
                Nnonb0     = numel(nonb0_pos);
                idx_nonb0  = [];
                for kl = 0:Nsl-1
                    idx_nonb0 = cat(1, idx_nonb0, kl*Nshells + nonb0_pos);
                end
        
                % Normalise and extract
                dwi_out = dwi_raw(:,:,:, idx_nonb0) ./ max(dwi_b0, askadam.epsilon);
        
                % Descriptors: one entry per non-b0 shell per Sl block
                bval_out   = repmat(bval_loop(~isb0),   Nsl, 1);
                ldelta_out = repmat(ldelta_loop(~isb0),  Nsl, 1);
                BDELTA_out = repmat(BDELTA_loop(~isb0),  Nsl, 1);
                te_out     = repmat(te_loop(~isb0),      Nsl, 1);
        
            else
                % NTE>1: keep b=0 (Sl0 only), no normalisation
                % b=0 shells contribute only to Sl0 block
                % non-b0 shells contribute to all Sl blocks
                % Output layout: [Sl0_b0(te1), Sl0_nonb0(te1),..., Sl0_b0(te2),...,
                %                 Sl2_nonb0(te1),..., Sl2_nonb0(te2),...]
        
                % Sl0 block: all shells (b=0 and non-b0), indices 1:Nshells
                % Sl2+ blocks: non-b0 only, indices kl*Nshells + nonb0_pos
                nonb0_pos = find(~isb0);
                Nnonb0    = numel(nonb0_pos);
                idx_out   = (1:Nshells).';   % Sl0 block: all shells
                for kl = 1:Nsl-1
                    idx_out = cat(1, idx_out, kl*Nshells + nonb0_pos);
                end
        
                dwi_out = dwi_raw(:,:,:, idx_out);
        
                % Descriptors aligned with output:
                % Sl0 block: all shells (b=0 and non-b0)
                % Sl2+ blocks: non-b0 only
                bval_out   = cat(1, bval_loop,             repmat(bval_loop(~isb0),   Nsl-1, 1));
                ldelta_out = cat(1, ldelta_loop,           repmat(ldelta_loop(~isb0),  Nsl-1, 1));
                BDELTA_out = cat(1, BDELTA_loop,           repmat(BDELTA_loop(~isb0),  Nsl-1, 1));
                te_out     = cat(1, te_loop,               repmat(te_loop(~isb0),      Nsl-1, 1));
            end
        
            fprintf('done.\n');
        end

        function [dwi_out, bval_out] = get_Sl_all_no_normalise(this, dwi, bval, bvec, ldelta, BDELTA, te, lmax)
        % GET_SL_ALL_NO_NORMALISE  Compute rotationally invariant spherical mean
        %   signals for all acquisition shells without b=0 normalisation.
        %
        % Loops over unique TE groups. Within each TE group:
        %   - b=0 is computed as the mean across all b=0 volumes (pooled across
        %     all ldelta and BDELTA), since b=0 signal is isotropic and independent
        %     of gradient timing. Higher-order Sl (l>0) are zero at b=0.
        %   - Non-b0 shells are computed per unique (ldelta, BDELTA) group using
        %     Sl_no_normalise.
        %
        % Output 4th dimension layout:
        %   [Sl0_shell1, ..., Sl0_shellN, Sl2_shell1, ..., Sl2_shellN, ...]
        %   where shells are ordered by the (te, ldelta, BDELTA, bval) loop,
        %   matching the ordering of unique_shell_keepb0.
        %   N = total number of unique shells (b=0 once per TE, non-b0 per group).
        %
        % Input
        % -----
        %   dwi     : 4D DWI data [x, y, z, volumes]
        %   bval    : 1D b-value vector, one per volume [ms/um^2]
        %   bvec    : 2D gradient directions [3, volumes]
        %   ldelta  : 1D gradient pulse duration, one per volume [ms]
        %   BDELTA  : 1D diffusion time, one per volume [ms]
        %   te      : 1D echo time, one per volume [ms]
        %   lmax    : maximum spherical harmonic order
        %
        % Output
        % ------
        %   dwi_out  : 4D rotationally invariant signal [x, y, z, Nshells*(lmax/2+1)]
        %   bval_out : 1D b-values of output shells in loop order [Nshells x 1]

        if isempty(te);      te     = zeros(size(bval)); end
        if isempty(ldelta);  ldelta = zeros(size(bval)); end
        if isempty(BDELTA);  BDELTA = zeros(size(bval)); end
        if isscalar(te);     te     = ones(size(bval))  * te;    end
        if isscalar(ldelta); ldelta = ones(size(bval)) * ldelta; end
        if isscalar(BDELTA); BDELTA = ones(size(bval)) * BDELTA; end
        
            dims = size(dwi, 1:3);
            Nsl  = lmax/2 + 1;
        
            Sl_blocks = {};
            bval_out  = [];
        
            te_unique = unique(te);
            for kt = 1:numel(te_unique)
        
                idx_te    = find(te == te_unique(kt));
                idx_b0_te = idx_te(bval(idx_te) == 0);
        
                % --- b=0: pool all b=0 volumes within this TE group ---
                % Sl0 = mean across all b=0 directions (isotropic, no SH decomposition needed)
                % Sl_l = 0 for l > 0 (no angular structure at b=0)
                if ~isempty(idx_b0_te)
                    dwi_b0_Sl0 = mean(dwi(:,:,:,idx_b0_te), 4);   % [x,y,z]
        
                    % Build 5D block [x,y,z, 1, Nsl]: Sl0=mean, Sl2...Sl_lmax=0
                    dwi_b0_5D            = zeros([dims, 1, Nsl], 'like', dwi_b0_Sl0);
                    dwi_b0_5D(:,:,:,1,1) = dwi_b0_Sl0;            % Sl0
                    % Sl2, Sl4, ... remain zero
        
                    Sl_blocks{end+1} = dwi_b0_5D;                 %#ok<AGROW>
                    bval_out         = cat(1, bval_out, 0);
                end
        
                % --- Non-b0 shells: one entry per (TE, ldelta, BDELTA, bval) ---
                ldelta_unique = unique(ldelta(idx_te));
                for klde = 1:numel(ldelta_unique)
        
                    idx_ldel      = intersect(find(ldelta == ldelta_unique(klde)), idx_te);
                    BDELTA_unique = unique(BDELTA(idx_ldel));
        
                    for kBDE = 1:numel(BDELTA_unique)
        
                        % Select non-b0 volumes in this (te, ldelta, BDELTA) group
                        idx_group = intersect(find(BDELTA == BDELTA_unique(kBDE)), idx_ldel);
                        idx_nonb0 = idx_group(bval(idx_group) > 0);
        
                        if isempty(idx_nonb0); continue; end
        
                        bval_group = bval(idx_nonb0);
                        bvec_group = bvec(:, idx_nonb0);
        
                        [dwi_Sl, b_unique] = this.Sl_no_normalise( ...
                            dwi(:,:,:,idx_nonb0), bval_group, bvec_group, lmax);
        
                        Nshells_group = numel(b_unique);
                        dwi_Sl_5D     = reshape(dwi_Sl, [dims, Nshells_group, Nsl]);
        
                        Sl_blocks{end+1} = dwi_Sl_5D;             %#ok<AGROW>
                        bval_out         = cat(1, bval_out, b_unique(:));
        
                    end
                end
            end
        
            % Concatenate all groups along shell dimension (dim 4)
            % Result: [x, y, z, Nshells_total, Nsl]
            dwi_5D = cat(4, Sl_blocks{:});
        
            % Reshape to [x, y, z, Nshells_total * Nsl] with layout:
            % [Sl0_shell1..Sl0_shellN | Sl2_shell1..Sl2_shellN | ...]
            Nshells_total = size(dwi_5D, 4);
            dwi_out = reshape(dwi_5D, [dims, Nsl * Nshells_total]);
        
        end

        function [dwi,b_all] = get_Sl_all(this,dwi,bval,bvec,ldelta,BDELTA,lmax)
            % [bval_sorted,ldelta_sorted,BDELTA_sorted] = this.unique_shell(bval,ldelta,BDELTA);
            % dims    = size(dwi);
            % tmp     = size([dims(1:3) size(bval_sorted)]);
            dims = size(dwi);
            tmp     = [];
            b_all   = [];
            % find unique little delta
            ldelta_unique   = unique(ldelta);
            for kldet = 1:numel(ldelta_unique)
                % for each little delta, find unique big delta
                idx_ldel    = find(ldelta == ldelta_unique(kldet));
                BDELTA_unique = unique(BDELTA(idx_ldel));
                for kBDE = 1:numel(BDELTA_unique)
                
                    % for each little delta and big delta, find unique b-values
                    idx_BDEL= intersect(find(BDELTA == BDELTA_unique(kBDE)),idx_ldel);
                    
                    bval_tmp            = bval(idx_BDEL);
                    bvec_tmp            = bvec(:,idx_BDEL);
                    [dwi_Sl,b_unique]   = this.Sl(dwi(:,:,:,idx_BDEL),bval_tmp,bvec_tmp,lmax);
                    dwi_Sl              = reshape(dwi_Sl,[dims(1:3) size(dwi_Sl,4)/(lmax/2+1) lmax/2+1]);
                    b_all = cat(2,b_all,b_unique);
                    tmp = cat(4,tmp,dwi_Sl);
                end
            end
            dwi = reshape(tmp,[dims(1:3), size(tmp,4)*size(tmp,5)]);
        end
        
        % compute rotational invariant DWI images
        function [dwi_Sl,bval_unique] = Sl(this,dwi,bval,bvec,lmax)
        % [dwi_sh,bval_unique] = Slm(this,dwi,bval,bvec,lmax)
        % Input
        % -----------
        % dwi        : 4D DWI data, [sx,sy,sz,sg]
        % bval       : 1D b-value vector
        % bvec       : 2D b-vector (gradient directions)
        % lmax       : Spherical Harmonic Order
        %
        % Output
        % -----------
        % dwi_sh     : 4D Rotational invariant images, [sx,sy,sz,slm]
        % bval_unique: unique b-value (no b0)
        %

            if size(bvec,1) == 3
                bvec = bvec.';
            end
            
            Nsh  = floor(lmax/2) + 1; 
            % get image size
            dims = size(dwi);
            % get unique bval;
            bval_unique = unique(bval);
            % get b=0 data for normalisation
            ind_b0 = bval == 0;
            if sum(ind_b0==0)==numel(ind_b0)
                dwi_b0      = ones(dims(1:3));
                NuniqueB    = numel(bval_unique);
            else
                dwi_b0      = mean(dwi(:,:,:,ind_b0),4);
                NuniqueB    = numel(bval_unique)-1;
            end
            
            % compute rotational invariant signal
            dwi_Sl       = zeros(numel(bval_unique)-1, Nsh, prod(dims(1:3)));
            counter = 0;
            for kb = 1:numel(bval_unique)
                
                if bval_unique(kb) ~= 0
                    counter = counter +1;
                    ind = bval == bval_unique(kb);
    
                    dwi_Sl(counter,:,:)   = this.SHrotinv(reshape( dwi(:,:,:,ind)./dwi_b0, prod(dims(1:3)),length(ind(ind>0))).', ...
                                                bvec(ind,:), lmax);
                end

            end
            
            dwi_Sl      = permute(reshape(dwi_Sl,[NuniqueB, Nsh, dims(1:3)]),[3 4 5 1 2]);
            dwi_Sl      = reshape(dwi_Sl(:,:,:,:,1:Nsh),[dims(1:3) NuniqueB*Nsh]);
            bval_unique = bval_unique(bval_unique ~= 0);
        end

        % compute rotational invariant DWI images
        function [dwi_Sl,bval_unique] = Sl_no_normalise(this,dwi,bval,bvec,lmax)
        % [dwi_sh,bval_unique] = Slm(this,dwi,bval,bvec,lmax)
        % Input
        % -----------
        % dwi        : 4D DWI data, [sx,sy,sz,sg]
        % bval       : 1D b-value vector
        % bvec       : 2D b-vector (gradient directions)
        % lmax       : Spherical Harmonic Order
        %
        % Output
        % -----------
        % dwi_sh     : 4D Rotational invariant images, [sx,sy,sz,slm]
        % bval_unique: unique b-value (no b0)
        %

            if size(bvec,1) == 3
                bvec = bvec.';
            end
            
            Nsh  = floor(lmax/2) + 1; 
            % get image size
            dims = size(dwi);
            % get unique bval;
            bval_unique = unique(bval);
            
            % compute rotational invariant signal
            dwi_Sl       = zeros(numel(bval_unique), Nsh, prod(dims(1:3)));
            for kb = 1:numel(bval_unique)

                % counter = counter +1;
                ind = bval == bval_unique(kb);

                dwi_Sl(kb,:,:)   = this.SHrotinv(reshape( dwi(:,:,:,ind), prod(dims(1:3)),length(ind(ind>0))).', ...
                                            bvec(ind,:), lmax);
            end
            
            dwi_Sl      = permute(reshape(dwi_Sl,[numel(bval_unique), Nsh, dims(1:3)]),[3 4 5 1 2]);
            dwi_Sl      = reshape(dwi_Sl(:,:,:,:,1:Nsh),[dims(1:3) numel(bval_unique)*Nsh]);
            
        end

        function F = SHrotinv(this, S, g, lmax)
            S       = cat(1, S, S);
            g       = cat(1, g,-g);
            dirs    = this.cart2sph_incl(g);
            Fnm     = leastSquaresSHT(lmax, S, dirs, 'real', []);
            nL      = floor(lmax)/2;
            F       = zeros(nL+1,size(S,2));
            IL      = @(l) l^2 + 2*l + 1;
            for i = 0:2:lmax
                list = IL(i-1)+1 : IL(i);
                F(i/2+1,:) = sqrt(sum(abs(Fnm(list,:)).^2,1))/sqrt(4*pi*(2*i+1));
            end
        end

    end

    methods(Static)

        function pl = WatsonSHexact(k)
            k = k(:).';
            p2 = 1/4*(3./sqrt(k)./dawson(sqrt(k)) -2 -3./k);
            p4 = 1/32./k.^2.*(105 + 12*k.*(5+k) + 5*sqrt(k).*(2*k-21)./dawson(sqrt(k)));
            pl = [ones(1,numel(k)); p2; p4];
        end
        
        % correct bvals
        function bval = RectifyBVal(bval,bval_target)

            if nargin < 2
                % round up to the 10^1 digit
                bval = round(bval/10)*10;
            else
                % match to the closest bval if it's available
                cost = abs(bval(:) - bval_target(:).');
                [~,ind] = min(cost,[],2);
                bval = bval_target(ind);
                
            end
            bval = bval / 1e3; % um2/ms

        end
        
        function bval_corr = rectify_bval_v2(bvals,thres)

            % maxi difference
            if nargin < 2
                thres = 0.05; % 5%
            end
            
            % round up bval
            bvals = round(bvals);
            
            % count occurance
            [gc, gr] = groupcounts(bvals');
            
            % grouping
            g = zeros(size((gr)));
            counter = 0;
            for k = 1:numel(gr)-1
                bw = thres*gr(k+1);
                if (gr(k+1) - gr(k)) > bw
                    counter = counter + 1;
                    g(k)    = counter;
                else
                    g(k)    = counter;
                end
            end
            g = circshift(g,1);
            % find unique group
            m = unique(g);
            
            % find unique b-values
            b_unique = zeros(numel(m),1);
            for k = 1:numel(m)
            
              idx = find( g == m(k) );
              [~,ii] = max(gc(idx));
              b_unique(k) = gr(idx(ii));
            
            end
            
            tmp_diff = abs(b_unique - bvals);
            
            [~,idx] = min(tmp_diff,[],1);
            
            bval_corr = b_unique(idx).';
        end

        % Cartiseian to spherical coordinate
        function dirs = cart2sph_incl(g)
            [azi, elev] = cart2sph(g(:,1),g(:,2),g(:,3));
            incl = pi/2-elev;
            dirs = [azi incl];
        end

        % get unique non-zero b-values for each little delta and big delta
        function [bval_sorted,ldelta_sorted,BDELTA_sorted] = unique_shell(bval,ldelta,BDELTA)
            bval_sorted     = [];
            ldelta_sorted   = [];
            BDELTA_sorted   = [];
            
            % find unique little delta
            ldelta_unique   = unique(ldelta);
            for klde = 1:numel(ldelta_unique)
                
                % for each little delta, find unique big delta
                idx_ldel    = find(ldelta == ldelta_unique(klde));
                BDELTA_unique = unique(BDELTA(idx_ldel));
                for kBDE = 1:numel(BDELTA_unique)
                
                    % for each little delta and big delta, find unique b-values
                    idx_BDEL= intersect(find(BDELTA == BDELTA_unique(kBDE)),idx_ldel);

                    b_unique = unique(bval(idx_BDEL));
                    b_unique = b_unique(b_unique>0);
                    
                    bval_sorted     = cat(2,bval_sorted,b_unique);
                    ldelta_sorted   = cat(2,ldelta_sorted,ones(size(b_unique))*ldelta_unique(klde));
                    BDELTA_sorted   = cat(2,BDELTA_sorted,ones(size(b_unique))*BDELTA_unique(kBDE));
                end
            
            end
            bval_sorted = bval_sorted(:);
            ldelta_sorted = ldelta_sorted(:);
            BDELTA_sorted = BDELTA_sorted(:);

        end

        % % UNIQUE_SHELL_KEEPB0  Extract unique acquisition shells from a dMRI protocol,
        % %                      preserving the grouping structure across acquisition parameters.
        % function [bval_sorted,ldelta_sorted,BDELTA_sorted,te_sorted] = unique_shell_keepb0(bval,ldelta,BDELTA,te,discardB0multiTE)
        % %
        % % Description
        % % -----------
        % % Identifies all unique diffusion shells defined by combinations of b-value,
        % % gradient pulse duration (little delta), diffusion time (big delta), and echo
        % % time. Shells are deduplicated within a nested hierarchy: unique TEs -> unique
        % % little deltas -> unique big deltas -> unique b-values. Unlike a simple unique()
        % % call, b=0 is retained once per unique (TE, ldelta, BDELTA) combination rather
        % % than being collapsed globally, which is important for models where the b=0
        % % signal varies across diffusion times or echo times.
        % %
        % % Syntax
        % % ------
        % %   [bval_sorted, ldelta_sorted, BDELTA_sorted, te_sorted] = ...
        % %       unique_shell_keepb0(bval, ldelta, BDELTA, te)
        % %
        % %   [bval_sorted, ldelta_sorted, BDELTA_sorted, te_sorted] = ...
        % %       unique_shell_keepb0(bval, ldelta, BDELTA, te, discardB0multiTE)
        % %
        % % Input
        % % -----
        % %   bval            : 1D vector, b-values [ms/um^2]
        % %   ldelta          : 1D vector, gradient pulse duration little delta [ms]
        % %                     Pass [] to treat all volumes as having the same little delta
        % %   BDELTA          : 1D vector, diffusion time big delta [ms]
        % %                     Pass [] to treat all volumes as having the same big delta
        % %   te              : 1D vector, echo time [ms]
        % %                     Pass [] or omit to treat all volumes as having the same TE
        % %   discardB0multiTE: logical scalar, whether to discard b=0 entries when only
        % %                     a single unique TE exists in the protocol (default: false)
        % %                     When true and the protocol has only one TE, b=0 is removed
        % %                     from the shell list since it carries no TE-dependent
        % %                     information and would otherwise appear redundantly once per
        % %                     unique (ldelta, BDELTA) combination
        % %
        % % Output
        % % ------
        % %   bval_sorted     : column vector, b-values of unique shells
        % %   ldelta_sorted   : column vector, little delta of unique shells [ms]
        % %   BDELTA_sorted   : column vector, big delta of unique shells [ms]
        % %   te_sorted       : column vector, echo time of unique shells [ms]
        % %
        % % Notes
        % % -----
        % %   - Output vectors are the same length and index-aligned: the k-th entry of
        % %     each output vector together describes the k-th unique shell
        % %   - Shells are ordered by the acquisition hierarchy (TE, ldelta, BDELTA, bval)
        % %     rather than by b-value magnitude
        % %   - If ldelta, BDELTA, or te are empty or omitted, they are treated as zero
        % %     (i.e. all volumes share the same value for that parameter)
        % %
        % % Example
        % % -------
        % %   % Protocol with two diffusion times and b=0 at each
        % %   bval   = [0 0 1 2 0 0 1 2];
        % %   ldelta = [10 10 10 10 20 20 20 20];
        % %   BDELTA = [20 20 20 20 40 40 40 40];
        % %   te     = ones(1,8) * 80;
        % %   [b, ld, BD, t] = unique_shell_keepb0(bval, ldelta, BDELTA, te);
        % %   % Returns 6 shells: b=0,1,2 for ldelta=10 and b=0,1,2 for ldelta=20
        % %
        % % Author
        % % ------
        % %   Kwok-Shing Chan (kchan2@mgh.harvard.edu)
        % 
        %     if nargin < 4; te              = [];    end
        %     if nargin < 5; discardB0multiTE = false; end
        % 
        %     if isempty(te);     te     = ones(size(bval));  end
        %     if isempty(ldelta); ldelta = zeros(size(bval)); end
        %     if isempty(BDELTA); BDELTA = zeros(size(bval)); end
        % 
        %     bval_sorted   = [];
        %     ldelta_sorted = [];
        %     BDELTA_sorted = [];
        %     te_sorted     = [];
        % 
        %     te_unique    = unique(te);
        %     hasMultiTE   = numel(te_unique) > 1;
        % 
        %     for kt = 1:numel(te_unique)
        % 
        %         idx_te        = find(te == te_unique(kt));
        %         ldelta_unique = unique(ldelta(idx_te));
        % 
        %         for klde = 1:numel(ldelta_unique)
        % 
        %             idx_ldel      = intersect(find(ldelta == ldelta_unique(klde)), idx_te);
        %             BDELTA_unique = unique(BDELTA(idx_ldel));
        % 
        %             for kBDE = 1:numel(BDELTA_unique)
        % 
        %                 idx_BDEL = intersect(find(BDELTA == BDELTA_unique(kBDE)), idx_ldel);
        %                 b_unique = unique(bval(idx_BDEL));
        % 
        %                 % Optionally discard b=0 when only a single TE exists,
        %                 % since in that case b=0 carries no TE-dependent information
        %                 % and retaining it per (ldelta, BDELTA) combination is redundant
        %                 if discardB0multiTE && ~hasMultiTE
        %                     b_unique = b_unique(b_unique > 0);
        %                 end
        % 
        %                 if isempty(b_unique); continue; end
        % 
        %                 bval_sorted   = cat(1, bval_sorted,   b_unique(:));
        %                 ldelta_sorted = cat(1, ldelta_sorted, ones(numel(b_unique),1) * ldelta_unique(klde));
        %                 BDELTA_sorted = cat(1, BDELTA_sorted, ones(numel(b_unique),1) * BDELTA_unique(kBDE));
        %                 te_sorted     = cat(1, te_sorted,     ones(numel(b_unique),1) * te_unique(kt));
        %             end
        %         end
        %     end
        % 
        % end
        % 
        function [bval_sorted,ldelta_sorted,BDELTA_sorted,te_sorted] = unique_shell_keepb0(bval,ldelta,BDELTA,te,discardB0)
        % UNIQUE_SHELL_KEEPB0  Extract unique acquisition shells from a dMRI protocol,
        %                      preserving the grouping structure across acquisition parameters.
        %
        % Description
        % -----------
        % Identifies all unique diffusion shells defined by combinations of b-value,
        % gradient pulse duration (little delta), diffusion time (big delta), and echo
        % time. Shells are deduplicated within a nested hierarchy: unique TEs -> unique
        % little deltas -> unique big deltas -> unique b-values. b=0 is retained once
        % per unique TE (pooled across ldelta and BDELTA), since b=0 signal depends on
        % TE but not on gradient timing parameters.
        %
        % Syntax
        % ------
        %   [bval_sorted, ldelta_sorted, BDELTA_sorted, te_sorted] = ...
        %       unique_shell_keepb0(bval, ldelta, BDELTA, te)
        %
        %   [bval_sorted, ldelta_sorted, BDELTA_sorted, te_sorted] = ...
        %       unique_shell_keepb0(bval, ldelta, BDELTA, te, discardB0)
        %
        % Input
        % -----
        %   bval      : 1D vector, b-values [ms/um^2]
        %   ldelta    : 1D vector, gradient pulse duration little delta [ms]
        %               Pass [] to treat all volumes as having the same little delta
        %   BDELTA    : 1D vector, diffusion time big delta [ms]
        %               Pass [] to treat all volumes as having the same big delta
        %   te        : 1D vector, echo time [ms]
        %               Pass [] or omit to treat all volumes as having the same TE
        %   discardB0 : logical scalar, whether to discard b=0 entries when only
        %               a single unique TE exists in the protocol (default: false)
        %               When true and the protocol has only one TE, b=0 is removed
        %               since it carries no TE-dependent information and would
        %               otherwise appear redundantly once per unique (ldelta, BDELTA)
        %
        % Output
        % ------
        %   bval_sorted  : column vector, b-values of unique shells
        %   ldelta_sorted: column vector, little delta of unique shells [ms]
        %                  For b=0 entries: smallest ldelta within that TE group
        %   BDELTA_sorted: column vector, big delta of unique shells [ms]
        %                  For b=0 entries: smallest BDELTA for the smallest ldelta
        %                  within that TE group
        %   te_sorted    : column vector, echo time of unique shells [ms]
        %
        % Notes
        % -----
        %   - Output vectors are the same length and index-aligned: the k-th entry
        %     of each output vector together describes the k-th unique shell
        %   - b=0 appears once per unique TE, assigned to the first (smallest ldelta,
        %     then smallest BDELTA) encountered within that TE group
        %   - Non-b0 shells appear once per unique (TE, ldelta, BDELTA, bval)
        %   - Shells are ordered: b=0(te1), non-b0(te1), b=0(te2), non-b0(te2), ...
        %   - If ldelta, BDELTA, or te are empty or omitted, they are treated as
        %     zeros (i.e. all volumes share the same value for that parameter)
        %
        % Example
        % -------
        %   % Protocol with two diffusion times, b=0 should appear once per TE
        %   bval   = [0 0 1 2 0 0 1 2];
        %   ldelta = [10 10 10 10 20 20 20 20];
        %   BDELTA = [20 20 20 20 40 40 40 40];
        %   te     = ones(1,8) * 80;
        %   [b, ld, BD, t] = unique_shell_keepb0(bval, ldelta, BDELTA, te);
        %   % Returns 3 shells: b=0(ld=10,bd=20), b=1, b=2 for ldelta=10,BDELTA=20
        %   % and b=1, b=2 for ldelta=20, BDELTA=40. b=0 appears only once.
        %
        % Author
        % ------
        %   Kwok-Shing Chan (kchan2@mgh.harvard.edu)
        
            if nargin < 4; te        = [];    end
            if nargin < 5; discardB0 = false; end
        
            if isempty(te);     te     = ones(size(bval));  end
            if isempty(ldelta); ldelta = zeros(size(bval)); end
            if isempty(BDELTA); BDELTA = zeros(size(bval)); end
        
            bval_sorted   = [];
            ldelta_sorted = [];
            BDELTA_sorted = [];
            te_sorted     = [];
        
            te_unique  = unique(te);
            hasMultiTE = numel(te_unique) > 1;
        
            for kt = 1:numel(te_unique)
        
                idx_te = find(te == te_unique(kt));
        
                % --- b=0: add once per TE group ---
                % assign to smallest ldelta, then smallest BDELTA within that TE
                has_b0 = any(bval(idx_te) == 0);
                if has_b0 && ~(discardB0 && ~hasMultiTE)
                    ld_first  = min(ldelta(idx_te));
                    idx_ld1   = intersect(find(ldelta == ld_first), idx_te);
                    bd_first  = min(BDELTA(idx_ld1));
        
                    bval_sorted   = cat(1, bval_sorted,   0);
                    ldelta_sorted = cat(1, ldelta_sorted,  ld_first);
                    BDELTA_sorted = cat(1, BDELTA_sorted,  bd_first);
                    te_sorted     = cat(1, te_sorted,      te_unique(kt));
                end
        
                % --- non-b0 shells: one entry per (TE, ldelta, BDELTA, bval) ---
                ldelta_unique = unique(ldelta(idx_te));
                for klde = 1:numel(ldelta_unique)
        
                    idx_ldel      = intersect(find(ldelta == ldelta_unique(klde)), idx_te);
                    BDELTA_unique = unique(BDELTA(idx_ldel));
        
                    for kBDE = 1:numel(BDELTA_unique)
        
                        idx_BDEL = intersect(find(BDELTA == BDELTA_unique(kBDE)), idx_ldel);
                        b_unique = unique(bval(idx_BDEL));
                        b_unique = b_unique(b_unique > 0);   % exclude b=0, handled above
        
                        if isempty(b_unique); continue; end
        
                        bval_sorted   = cat(1, bval_sorted,   b_unique(:));
                        ldelta_sorted = cat(1, ldelta_sorted,  ones(numel(b_unique),1) * ldelta_unique(klde));
                        BDELTA_sorted = cat(1, BDELTA_sorted,  ones(numel(b_unique),1) * BDELTA_unique(kBDE));
                        te_sorted     = cat(1, te_sorted,      ones(numel(b_unique),1) * te_unique(kt));
                    end
                end
            end
        end

        % % get unique b-values for each little delta and big delta
        % function [bval_sorted,ldelta_sorted,BDELTA_sorted,te_sorted] = unique_shell_keepb0(bval,ldelta,BDELTA,te)
        %     if nargin < 4
        %         te = ones(size(bval));
        %     end
        %     bval_sorted     = [];
        %     ldelta_sorted   = [];
        %     BDELTA_sorted   = [];
        %     te_sorted       = [];
        % 
        %     if isempty(ldelta)
        %         ldelta = zeros(size(bval));
        %     end
        %     if isempty(BDELTA)
        %         BDELTA = zeros(size(bval));
        %     end
        %     if isempty(te)
        %         te = zeros(size(bval));
        %     end
        % 
        %     % find unique te
        %     te_unique = unique(te);
        %     for kt = 1:numel(te_unique)
        % 
        %         % for each little te, find unique little delta
        %         idx_te = find(te == te_unique(kt));
        %         % find unique little delta
        %         ldelta_unique   = unique(ldelta(idx_te));
        % 
        %         for klde = 1:numel(ldelta_unique)
        % 
        %             % for each little delta, find unique big delta
        %             idx_ldel        = intersect(find(ldelta == ldelta_unique(klde)), idx_te);
        %             BDELTA_unique   = unique(BDELTA(idx_ldel));
        %             for kBDE = 1:numel(BDELTA_unique)
        % 
        %                 % for each little delta and big delta, find unique b-values
        %                 idx_BDEL= intersect(find(BDELTA == BDELTA_unique(kBDE)),idx_ldel);
        % 
        %                 b_unique = unique(bval(idx_BDEL));
        %                 % b_unique = b_unique(b_unique>0);
        % 
        %                 bval_sorted     = cat(2,bval_sorted,b_unique);
        %                 ldelta_sorted   = cat(2,ldelta_sorted,ones(size(b_unique))*ldelta_unique(klde));
        %                 BDELTA_sorted   = cat(2,BDELTA_sorted,ones(size(b_unique))*BDELTA_unique(kBDE));
        %                 te_sorted       = cat(2,te_sorted,ones(size(b_unique))*te_unique(kt));
        %             end
        %         end
        % 
        %     end
        %     bval_sorted     = bval_sorted(:);
        %     ldelta_sorted   = ldelta_sorted(:);
        %     BDELTA_sorted   = BDELTA_sorted(:);
        %     te_sorted       = te_sorted(:);
        % 
        % end
        
        % permute the 4th dimension of input data to 1st dimension
        function data = permute_dwi_dimension(data)
            if isstruct(data)
                fn = fieldnames(data);
                for k = 1:numel(fn)   
                    data.(fn{k}) = permute(data.(fn{k}),[4 1 2 3]);
                end
            else
                data = permute(data,[4 1 2 3]);
            end
        end

        % permute the 1st dimension of input data to 4th dimension (i.e. undo permute_dwi_dimension)
        function data = unpermute_dwi_dimension(data)
            if isstruct(data)
                fn = fieldnames(data);
                for k = 1:numel(fn)   
                    data.(fn{k}) = permute(data.(fn{k}),[2 3 4 1]);
                end
            else
                data = permute(data,[2 3 4 1]);
            end
        end
    
        % vectorise 4D image to 2D with the same last dimention
        function [data, mask_idx] = vectorise_4Dto2D(data,mask)

            dims = size(data,[1 2 3]);

            if nargin < 2
                mask = ones(dims);
            end

            % vectorise data
            data        = reshape(data,prod(dims),size(data,4));
            mask_idx    = find(mask>0);
            data        = data(mask_idx,:);

        end
    
    end

end

