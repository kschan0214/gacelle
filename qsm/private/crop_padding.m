function x = crop_padding(x, extraData)
% Undo check_padsize. Leaves x untouched if no padding was applied.
% Works for any number of trailing dimensions.

    if ~isfield(extraData,'padPre') || all([extraData.padPre extraData.padPost] == 0)
        return
    end

    p  = extraData.padPre;
    m0 = extraData.matrixSize0;

    x = x(p(1)+1 : p(1)+m0(1), ...
          p(2)+1 : p(2)+m0(2), ...
          p(3)+1 : p(3)+m0(3), :, :);

end