
layers = get_lenet();
load lenet.mat
% load data
% Change the following value to true to load the entire dataset.
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
batch_size = 64;
 
 
layers{1}.batch_size = 1;
img = xtest(:, 1);
img = reshape(img, 28, 28);
imshow(img')
 
%[cp, ~, output] = conv_net_output(params, layers, xtest(:, 1), ytest(:, 1));
output = convnet_forward(params, layers, xtest(:, 1));
output_1 = reshape(output{1}.data, output{1}.width, output{1}.height);
% Fill in your code here to plot the features.
second_layer = reshape(output{2}.data, output{2}.width, output{2}.height, output{2}.channel);
third_layer = reshape(output{3}.data, output{3}.width, output{3}.height, output{3}.channel);

normalize_layer = true;
for inum=2:3
    figure;
    layer_data = reshape(output{inum}.data, output{inum}.width, output{inum}.height, output{inum}.channel);
    for ii = 1:20
       subplot(4, 5,ii);
       m = layer_data(:, :, ii)';
       if normalize_layer
           min1 = min(m, [], 'all');
           max1 = max(m, [], 'all');
           m = uint8(255 .* ((double(m)-min1)) ./ (max1-min1));
       end
       imshow(m);
    end
end
