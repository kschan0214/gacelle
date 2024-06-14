classdef DWIutility

    properties (GetAccess = public, SetAccess = protected)
        % b;
        % Delta;
    end

    methods

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

