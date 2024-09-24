function st = cell2str(cellStr)
    cellStr= cellfun(@(x){[x ',']},cellStr);  % Add ',' after each string.
    st = cat(2,cellStr{:});  % Convert to string
    st(end) = [];  % Remove last ','
end