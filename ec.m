addpath('../matlab/');
layers = get_lenet();
load lenet.mat;

ims = {rgb2gray(imread('../images/image1.JPG')) ...
    rgb2gray(imread('../images/image2.JPG')) ...
    rgb2gray(imread('../images/image3.png')) ...
    rgb2gray(imread('../images/image4.jpg'))};

pads = [5 45; 5 20; 1 30; 1 5];
labels = {[1 2 3 4 5 6 7 8 9 0], ...
    [1 2 3 4 5 6 7 8 9 0], ...
    [6 0 6 2 4], ...
    [7 0 9 3 1 6 7 2 6 1 3 9 6 4 1 4 2 0 0 ...
    5 4 4 7 3 1 0 2 5 5 1 7 7 4 9 1 7 4 2 9 1 ...
    5 3 4 0 2 -1 9 4 4 1 1 ]};
total = 0;
accurate = 0;
for i=1:4
    I = ims{i};
    image_pad = pads(i, 1);
    input_pad = pads(i, 2);
    image_labels = labels{i};
    %I = I>(max(I, [], 'all')-min(I, [], 'all'))/1.5;
    I = imbinarize(I,graythresh(I));
    mask = zeros(size(I));
    mask(1:end,1:end) = 1;
    bw = 1 - activecontour(I,mask);
    bwc = bwconncomp(bw);
    components = bwc.PixelIdxList;
    figure;
    for cc=1:bwc.NumObjects
       ccc = components(cc);
       cc_array = ccc{1};
       if size(cc_array, 1) < 2
           continue
       end
       x_coords = zeros(1, size(cc_array, 1));
       y_coords = zeros(1, size(cc_array, 1));
       for point=1:size(cc_array, 1)
           [x1 y1] = ind2sub(size(bw), cc_array(point));
           x_coords(point) = x1;
           y_coords(point) = y1;
       end
       smallest_y = max(min(y_coords)-image_pad, 1);
       smallest_x = max(min(x_coords)-image_pad, 1);
       largetst_y = min(max(y_coords)+image_pad, size(bw, 2));
       largest_x = min(max(x_coords)+image_pad, size(bw, 1));
       image_segment = bw(smallest_x:largest_x, smallest_y:largetst_y);
       image_segment = padarray(image_segment,[input_pad input_pad],0,'both');
       image_segment = imresize(image_segment,[28 28]);
       %image_segment = imsharpen(image_segment);
       layers{1}.batch_size = 1;
       img = reshape(image_segment', 784, 1);
       [output, P] = convnet_forward(params, layers, img);
       class_probabilities = P(:, 1);
       [value prediction] = max(class_probabilities);
       if i < 3
           subplot(2, 5, cc);
           imshow(image_segment);
       else
           if i == 3
            subplot(1, 6,cc);
            imshow(image_segment);
           else
            subplot(6, 10,cc);
            imshow(image_segment);
           end
       end
       actual = image_labels(cc);
       fprintf("%d(%d), ", prediction-1, actual);
       if rem(cc, 10) == 0
           fprintf("\n");
       end
       if actual > -1 % filtering out the noise
           total = total + 1;
           if prediction-1 == actual
               accurate = accurate + 1;
           end
       end
    end
    fprintf("\n");
end
fprintf("\nReal case accuracy: %.1f%%\n", accurate * 100 /total);