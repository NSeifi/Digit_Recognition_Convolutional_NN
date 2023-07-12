function [output] = conv_layer_forward(input, layer, param)
% Conv layer forward
% input: struct with input data
% layer: convolution layer struct
% param: weights for the convolution layer

% output: 

h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;
num = layer.num;
% resolve output shape
h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;

assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')
input_n.height = h_in;
input_n.width = w_in;
input_n.channel = c;

%% Fill in the code
% Iterate over the each image in the batch, compute response,
% Fill in the output datastructure with data, and the shape. 
result = zeros(w_out * h_out * num, batch_size);
for n = 1:batch_size
i_data = reshape(input.data(:, n), w_in, h_in, c);
responses = zeros(w_out, h_out, num);
    for f=1:num
        filters = param.w(:, f);
        b_f = param.b(f);
        filter = reshape(filters, k, k, c);
        c_data = padarray(i_data,[pad pad],0,'both');
          for i=1:stride:w_in+pad
            for j=1:stride:h_in+pad
                ri = floor(i/stride+0.999);
                rj = floor(j/stride+0.999);
                if ri > w_out || rj > h_out
                    break;
                end
                pi = min(i+k-1, w_in+2*pad);
                pj = min(j+k-1, h_in+2*pad);
                responses(ri, rj, f) = sum(c_data(i:pi, j:pj, :) .* filter, 'all') + b_f;
            end
          end
    end
    result (:,n) = reshape (responses, w_out * h_out * num, 1);
    end
output.height = w_out;
output.width = h_out;
output.channel = num;
output.batch_size = batch_size;
output.data = result;

end

