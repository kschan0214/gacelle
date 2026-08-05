function [r2s,M0] = R2star_trapezoidal(img,te)
% Fast closed-form R2* from the trapezoidal integral of the decay
% curve. Approximate, but adequate for binning and masking.
%
% Produces Inf/NaN wherever the integral is zero (background air,
% dead channels). Callers must sanitise: see thres_R2s handling in
% compute_adaptive_preconditioner.

    % disgard phase information
    img = double(abs(img));
    te  = double(te);
    
    % Trapezoidal approximation of integration
    temp = 0;
    for k = 1:size(img,4)-1
        temp = temp + 0.5*(img(:,:,:,k)+img(:,:,:,k+1))*(te(k+1)-te(k));
    end
    
    % very fast estimation
    r2s = (img(:,:,:,1)-img(:,:,:,end)) ./ temp;

    M0 = img(:,:,:,1) .*exp(r2s .* te(1));

end