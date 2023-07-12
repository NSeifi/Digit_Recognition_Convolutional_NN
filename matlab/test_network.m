%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat

%% Testing the network
% Modify the code to get the confusion matrix
predictions = zeros(1, size(ytest, 2));
for i=1:100:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    for batch_item=1:size(P, 2)
        class_probabilities = P(:, batch_item);
        [value prediction] = max(class_probabilities);
        predictions(i+batch_item-1) = prediction;
    end
end

fprintf("Accuracy : %.2f%%\n", (sum(ytest == predictions)/ size(ytest, 2))*100);
Conf_Mat = confusionmat(ytest, predictions);
fprintf('The confusion matrix:\n')
disp(Conf_Mat);
