function [input_od] = relu_backward(output, input, layer)

% Replace the following line with your implementation.
input_od = output.diff .* (output.data > 1);
end
