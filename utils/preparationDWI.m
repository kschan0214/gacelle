classdef preparationDWI

    properties (GetAccess = public, SetAccess = protected)
        % b;
        % Delta;
    end

    methods
        
        % compute rotational invariant DWI images
        function [dwi_sh,bval_unique] = Slm(this,dwi,bval,bvec,lmax)
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
            % use 2 higher order for computation
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
            dwi_sh       = zeros(numel(bval_unique)-1, Nsh, prod(dims(1:3)));
            counter = 0;
            for kb = 1:numel(bval_unique)
                
                if bval_unique(kb) ~= 0
                    counter = counter +1;
                    ind = bval == bval_unique(kb);
    
                    dwi_sh(counter,:,:)   = this.SHrotinv(reshape( dwi(:,:,:,ind)./dwi_b0, prod(dims(1:3)),length(ind(ind>0))).', ...
                                                bvec(ind,:), lmax);
                end

            end
            
            dwi_sh      = permute(reshape(dwi_sh,[NuniqueB, Nsh, dims(1:3)]),[3 4 5 1 2]);
            dwi_sh      = reshape(dwi_sh(:,:,:,:,1:Nsh),[dims(1:3) NuniqueB*Nsh]);
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

    end

end

