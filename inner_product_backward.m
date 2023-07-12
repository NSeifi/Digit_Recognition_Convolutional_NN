function [param_grad, input_od] = inner_product_backward(output, input, layer, param)

% Replace the following lines with your implementation.
batch_size = input.batch_size;
param_grad.b = zeros(size(param.b));
param_grad.w = zeros(size(param.w));

for n = 1:batch_size
   n_output = output.diff(:, n); % of size h_out
   n_input = input.data(:, n); % of size w_in * h_in * c
   param_grad.w = param_grad.w + n_input * n_output';
   param_grad.b = param_grad.b + n_output';
end
param_grad.w = param_grad.w ./ batch_size;
param_grad.b = param_grad.b ./ batch_size;
input_od = param.w * output.diff;

end
