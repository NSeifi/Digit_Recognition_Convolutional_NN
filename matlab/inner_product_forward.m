function [output] = inner_product_forward(input, layer, param)

d = size(input.data, 1);
k = size(input.data, 2); % batch size
n = size(param.w, 2);

% Replace the following line with your implementation.
output.height = n;
output.width = 1;
output.channel = 1;
output.batch_size = k;
output.data = (input.data' * param.w + repmat(param.b, k, 1))';

end
