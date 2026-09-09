
layers = parameters;
totalParams = 0;

fn = fieldnames(layers);

for i = 1:numel(fn)
    layer = layers.(fn{i});
    
    % if isprop(layer, 'Weights')
        totalParams = totalParams + numel(layer.Weights);
    % end
    
    % if isprop(layer, 'Bias')
        totalParams = totalParams + numel(layer.Bias);
    % end
end

disp(['Total number of learnable parameters: ', num2str(totalParams)]);


