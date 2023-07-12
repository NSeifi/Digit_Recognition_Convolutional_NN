function [output] = pooling_layer_forward(input, layer)

    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = batch_size;

    % Replace the following line with your implementation.
    result = zeros(w_out * h_out * c, batch_size);
for n = 1:batch_size
i_data = reshape(input.data(:, n), w_in, h_in, c);
responses = zeros(w_out, h_out, c);
    for f=1:c
        c_data = padarray(i_data(:, :, f),[pad pad],0,'both');
          for i=1+pad:stride:w_in+pad
            for j=1+pad:stride:h_in+pad
                ri = floor(i/stride+0.999);
                rj = floor(j/stride+0.999);
                if ri > w_out || rj > h_out
                    break;
                end
                pi = min(i+k-1, w_in+pad);
                pj = min(j+k-1, h_in+pad);
                responses(ri, rj, f) = max(c_data(i:pi, j:pj), [], 'all');
            end
          end
    end
    result (:,n) = reshape (responses, w_out * h_out * c, 1);
    end
output.height = w_out;
output.width = h_out;
output.channel = c;
output.batch_size = batch_size;
output.data = result;

end

