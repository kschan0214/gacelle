function st = cell2num2str(cellStr)
    cellStr= cellfun(@(x){[num2str(x) ',']},cellStr);  % Add ',' after each string.
    st = cat(2,cellStr{:});  % Convert to string
    st(end) = [];  % Remove last ','
end