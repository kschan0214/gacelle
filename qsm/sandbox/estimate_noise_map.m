        function [sigma, SNR] = estimate_noise_map( S, mask, winSize)
        %ESTIMATE_NOISE_MAP  Spatially varying noise SD from complex multi-echo data.
        %
        %   [sigma, SNR] = this.estimate_noise_map(S, mask)
        %   [sigma, SNR] = this.estimate_noise_map(S, mask, winSize)
        %
        % No fitting, no ROI, no background region required.
        %
        % Input
        % -----
        %   S       : complex multi-echo GRE, [x,y,z,nTE]
        %   mask    : ROI, used only to report SNR and to fill the output
        %   winSize : local window for the median, default 9 (voxels).
        %             Larger = more stable, less able to track g-factor
        %             structure. 7-11 is usually a reasonable range.
        %
        % Output
        % ------
        %   sigma   : [x,y,z] noise SD of ONE channel (real or imaginary),
        %             in the same units as S
        %   SNR     : [x,y,z] |S(TE1)| ./ sigma
        %
        % Method
        %   1. Work on real and imaginary channels separately. Each carries
        %      zero-mean Gaussian noise with the SAME sigma, so there is no
        %      Rician bias - which is exactly what corrupts magnitude-based
        %      estimates in the low-SNR voxels that matter most.
        %   2. Filter with a 2x2x2 alternating-sign kernel (the Haar HHH
        %      detail, = outer product of [1;-1] along each axis). This
        %      annihilates any trilinear function exactly, so smooth anatomy
        %      contributes nothing while white noise passes with gain
        %      sqrt(sum(h.^2)) = sqrt(8).
        %   3. Robust local scale: for zero-mean Gaussian d,
        %      median(|d|) = 0.6745*sigma_d, so sigma_d = 1.4826*median(|d|).
        %      The median ignores the small fraction of voxels where a vessel
        %      or tissue boundary breaks the trilinear assumption; a standard
        %      deviation would not.
        %   4. Pool across all 2*nTE channels before taking the local
        %      median. Thermal noise sigma is TE independent, so every echo
        %      and both channels are repeat measurements of the same
        %      quantity - this is where most of the stability comes from.
        %
        % CAVEATS - the estimate is biased LOW if the noise is spatially
        % correlated, because correlated noise is partly annihilated along
        % with the signal. This happens with:
        %   - zero-filled / interpolated k-space
        %   - partial Fourier with homodyne reconstruction
        %   - any k-space filtering (Hanning, Fermi)
        %   - denoising applied upstream (MPPCA, NORDIC)
        % If any of these apply, treat sigma as a lower bound. A quick check:
        % the residual RMS after fitting should be comparable to sigma. If
        % the residual is systematically BELOW it, either the fit is
        % overfitting or sigma is underestimated.
        %
        % If the data was coil combined with adaptive/SENSE weighting, sigma
        % IS genuinely spatially varying (g-factor) and this method tracks
        % it, which is the reason for estimating a map rather than a scalar.

            if nargin < 4 || isempty(winSize); winSize = 9; end

            matrixSize = size(S,1:3);
            nTE        = size(S,4);

            % 2x2x2 alternating-sign kernel; sum(h(:).^2) = 8
            v = [1; -1];
            h = reshape(kron(kron(v, v), v), 2, 2, 2);

            % ---- pool |detail| across real/imag and all echoes ----------
            absd = zeros([matrixSize, 2*nTE], 'single');
            k    = 0;
            for n = 1:nTE
                for c = 1:2
                    if c == 1
                        x = real(single(S(:,:,:,n)));
                    else
                        x = imag(single(S(:,:,:,n)));
                    end
                    k = k + 1;
                    absd(:,:,:,k) = abs(convn(x, h, 'same'));
                end
            end

            % single pooled volume: median over channels first, so one
            % noisy echo cannot dominate the local statistic
            absd = median(absd, 4);

            % ---- local median -> robust sigma ---------------------------
            % medfilt3 needs odd window
            w = winSize + (1 - mod(winSize,2));
            med = medfilt3(absd, [w w w], 'symmetric');

            sigma = 1.4826 * med ./ sqrt(sum(h(:).^2));     % sqrt(8)

            % ---- tidy up ------------------------------------------------
            % convn 'same' contaminates a one-voxel border; replace it
            valid = false(matrixSize);
            valid(2:end-1, 2:end-1, 2:end-1) = true;
            sigma(~valid) = NaN;
            sigma = fillmissing(sigma, 'nearest', 1);
            sigma = fillmissing(sigma, 'nearest', 2);
            sigma = fillmissing(sigma, 'nearest', 3);

            % guard against zeros (fully zero-filled regions)
            sigma = max(sigma, eps('single'));

            if nargout > 1
                SNR = abs(single(S(:,:,:,1))) ./ sigma;
            end

            if nargin > 2 && ~isempty(mask)
                fprintf('Noise SD in ROI : median %.4g, IQR [%.4g %.4g]\n', ...
                    median(sigma(mask)), prctile(sigma(mask),25), prctile(sigma(mask),75));
                if nargout > 1
                    fprintf('SNR at TE1      : median %.1f\n', median(SNR(mask)));
                end
            end

        end
