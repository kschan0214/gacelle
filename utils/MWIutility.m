classdef MWIutility

    properties (GetAccess = public, SetAccess = protected)

    end

    methods

        % function 
        % 
        % end

    end

    methods(Static)
        
        function [SF, img] = image_normalisation_ND(img,mask)
        % [scaleFactor, img] = mwi_image_normalisation(img, scaleMask)
        %
        % Input
        % --------------
        % img           : image to be normalised
        % scaleMask     : signal mask
        %
        % Output
        % --------------
        % SF            : scaling factor
        % img           : normalised images
        %
        % Description:
        %
        % Kwok-shing Chan @ DCCN
        % k.chan@donders.ru.nl
        % Date created: 16 Nov 2020
        % Date modified:
        %

            dims = size(img);

            % reshape to 4D image
            img_max = reshape(abs(img),[dims(1:3),prod(dims(4:end))]);
            img_max = max(img_max,[],4);

            % compute a signal mask if it is not provided
            if nargin < 2
                mask = img_max/prctile(img_max(:),95) > 0.15;
            end

            % determine the scaling factor
            SF = norm(img_max(mask>0)) / sqrt(length(find(mask>0)));
            
            img = img / SF;

        end
        
        function theta = AngleBetweenV1MapAndB0(v1,b0dir)
        % theta = AngleBetweenV1MapAndB0(v1,b0dir)
        %
        % Input
        % --------------
        % v1            : 4D fibre orientation map in vector form
        % b0dir         : 1D vector of B0 direction
        %
        % Output
        % --------------
        % theta         : 3D angle map, in rad
        %
        % Description:
        %
        % Kwok-shing Chan @ DCCN
        % k.chan@donders.ru.nl
        % Date created: 20 March 2019
        % Date last modified: 25 October 2019
        %
        %

        % replicate B0 direction to all voxels
        b0dirmap = permute(repmat(b0dir(:),1,size(v1,1),size(v1,2),size(v1,3)),[2 3 4 1]);
        % compute angle between B0 direction and fibre orientation
        theta = atan2(vecnorm(cross(v1,b0dirmap),2,4), dot(v1,b0dirmap,4));
        
        % make sure the angle is in range [0, pi/2]
        theta(theta> (pi/2)) = pi - theta(theta> (pi/2));
        
        end

        function [m0,mwf] = superfast_mwi_2m_mcr(img,te,fa,tr,t2s,t1,mask,b1map)
        % [m0,mwf] = superfast_mwi_2m_mcr(img,te,fa,tr,t2s,t1,mask,b1map)
        %
        % Input
        % --------------
        % img           : variable flip angle multi-echo GRE image, 5D [row,col,slice,TE,Flip angle]
        % te            : echo times in second
        % fa            : flip angle in degree
        % tr            : repetition time in second
        % t2s           : T2* of the two pools, in second, [T2sMW,T2sIEW], if empty
        %                 then default values for 3T will be used
        % t1            : T1 of the two pools, in second, [T1MW, T1IEW], if empty
        %                 then default values for 3T will be used
        % mask          : signal mask, (optional)
        % b1map         : B1 flip angel ratio map, (optional)
        %
        % Output
        % --------------
        % m0            : proton density of each pool, 4D [row,col,slice,pool]
        % mwf           : myelin water fraction map, range [0,1]
        %
        % Description:  Direct matrix inversion based on simple 2-pool model, i.e.
        %               S(te,fa) = E1 * M0 * E2s
        %               Useful to estimate initial starting points for MWI fitting
        %
        % Kwok-shing Chan @ DCCN
        % k.chan@donders.ru.nl
        % Date created: 12 Nov 2020
        % Date modified:
        %
        %
        % get size in all image dimensions
        dims(1) = size(img,1);
        dims(2) = size(img,2);
        dims(3) = size(img,3);
        
        % check input
        if isempty(t2s)
            t2s = [10e-3, 60e-3];   % 3T, [MW, IEW], in second
        end
        if isempty(t1)
            t1 = [234e-3, 1];       % 3T, [MW, IEW], in second
        end
        if nargin < 8
            b1map = ones(dims);
        end
        if nargin < 7
            mask = ones(dims);
        end
        
        % T2* decay matrix
        E2s1    = exp(-te(:).'/t2s(1));
        E2s2	= exp(-te(:).'/t2s(2));
        E2s     = [E2s1;E2s2];
        
        tmp = reshape(abs(img),prod(dims),length(te),length(fa));
        
        m0   = zeros([prod(dims),2]);
        for k = 1:prod(dims)
            if mask(k) ~= 0
                
                % T1-T2* signal
                temp = squeeze(tmp(k,:,:)).';
        
                % T1 steady-state matrix
                E11 = sind(fa(:)*b1map(k)) .* (1-exp(-tr/t1(1)))./(1-cosd(fa(:)*b1map(k))*exp(-tr/t1(1)));
                E12 = sind(fa(:)*b1map(k)) .* (1-exp(-tr/t1(2)))./(1-cosd(fa(:)*b1map(k))*exp(-tr/t1(2)));
                E1  = [E11,E12];
                
                % matrix inversion
                temp2 = (E1 \ temp) / E2s;
                
                % the diagonal components represent signal amplitude
                m0(k,:) =  diag(temp2);
            end
        
        end
        m0 = reshape(m0,[dims 2]);
        
        % compute MWF
        mwf = m0(:,:,:,1) ./ sum(m0,4);
        mwf(mwf<0)      = 0;
        mwf(isnan(mwf)) = 0;
        mwf(isinf(mwf)) = 0;
        
        m0(m0 < 0)      = 0;
        m0(isinf(m0))   = 0;
        m0(isnan(m0))   = 0;
        
        end

    end

end

