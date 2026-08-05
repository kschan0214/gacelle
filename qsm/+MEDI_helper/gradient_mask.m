function MG = gradient_mask( img, mask, voxelSize, grad, percentage)
    % adapted from MEDI toolbox

    if nargin < 6
        percentage = 0.9;
    end
    
    field_noise_level   = 0.01*max(img(:));
    MG                  = abs(grad(img.*(mask>0), voxelSize));
    denominator         = nnz(mask);
    numerator           = sum(MG(:)>field_noise_level);

    if  (numerator/denominator) > percentage
        while (numerator/denominator) > percentage
            field_noise_level   = field_noise_level*1.05;
            numerator           = sum(MG(:)>field_noise_level);
        end
    else
        while (numerator/denominator) < percentage
            field_noise_level   = field_noise_level*.95;
            numerator           = sum(MG(:)>field_noise_level);
        end
    end
    
    MG = MG <= field_noise_level;
end