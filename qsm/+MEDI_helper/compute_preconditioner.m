function P = compute_preconditioner(Ps,data,mask,method, R2s, thres_R2s)

    if nargin < 6
        thres_R2s = 30;
    end

    matrixSize = size(data,1:3);

    % ----- validate the preconditioner -----
    switch lower(method)
        % case 'auto'
        %     P           = this.compute_adaptive_preconditioner(data, extraData.img, mask);
        %     extraData.P = P;

        case 'empirical'
            P           = ones(matrixSize,'single');
            P(~mask)    = Ps;

        case 'none'
            P           = ones(matrixSize,'single');

        case 'r2s'

            if isempty(R2s)
                error('r2s requires input R2* map.');
            end

            P                   = ones(matrixSize,'single');
            P(R2s > thres_R2s)  = Ps;    % ICH / calcification
            
        case 'emp+r2s'

            if isempty(R2s)
                error('emp+r2s requires input R2* map.');
            end

            % Liu 2017 empirical + R2* preconditioner. The 30 Hz
            % cutoff is the value implied by the Liu 2020 sigmoid
            % reducing to this hard threshold (sigma1=1, sigma2=30,
            % s1=30, s2<<1). Verify against the 2017 paper before
            % quoting it.
            %
            % NOTE: R2s is not sanitised here, unlike in
            % compute_adaptive_preconditioner. Inf from the
            % trapezoidal fit passes the > 30 test, so pure-noise
            % voxels get labelled as strong sources. Apply the same
            % thres_R2s clamp before thresholding.
            P                           = ones(matrixSize,'single');
            P(~mask)                    = Ps;    % background
            P(mask & R2s > thres_R2s)   = Ps;    % ICH / calcification

    end

end