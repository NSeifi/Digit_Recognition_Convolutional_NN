% Network defintion
addpath('../matlab/');
layers = get_lenet();
load lenet.mat

%% Loading data
ims = {imresize(rgb2gray(imread('../images/2.JPG')),[28,28]) ...
    imresize(rgb2gray(imread('../images/3.JPG')),[28,28]) ...
    imresize(rgb2gray(imread('../images/4.JPG')),[28,28]) ...
    imresize(rgb2gray(imread('../images/5.JPG')),[28,28]) ...
    imresize(rgb2gray(imread('../images/7.JPG')),[28,28])};
labels = {[2], ...
    [3], ...
    [4], ...
    [5], ...
    [7]};
total = 0;
accurate = 0;
%% Testing the network
for i=1:5
    image_labels = labels{i};
    layers{1}.batch_size = 1;
    imd=im2double(ims{i});
    img = 1 - reshape(imd', 784, 1);
    [output, P] = convnet_forward(params, layers, img);
        class_probabilities = P(:, 1);
       [value prediction] = max(class_probabilities);
       prediction = prediction - 1 ;
       fprintf("%d(%d), ", prediction, image_labels);
    
           total = total + 1;
           if prediction == image_labels
               accurate = accurate + 1;
           end
      
    fprintf("\n");
end
fprintf("\nReal case accuracy: %.1f%%\n", accurate * 100 /total);
