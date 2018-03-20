clear
outputFolder = fullfile('./');
rootFolder = fullfile(outputFolder, 'Photo');
categories = {'gmh','Xiaoyi'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);
% determine the smallest amount of images in a category
minSetCount = min(tbl{:,2});
% Use partition method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');
% Notice that each set now has exactly the same number of images.
countEachLabel(imds);
cnnMatFile = fullfile(outputFolder, 'imagenet-caffe-alex.mat');
convnet = helperImportMatConvNet(cnnMatFile);
imds.ReadFcn = @(filename)readAndPreprocessImagef(filename);
[trainingSet, testSet] = splitEachLabel(imds, 0.6, 'randomize');
% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;
% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
featureLayer = 'fc7';
trainingFeatures = activations(convnet, trainingSet, featureLayer, ...
'MiniBatchSize', 32, 'OutputAs', 'columns');
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
% Extract test features using the CNN
testFeatures = activations(convnet, testSet, featureLayer, 'MiniBatchSize',32);
% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);
% Get the known labels
testLabels = testSet.Labels;
% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels)
% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))
% Display the mean accuracy
mean(diag(confMat))
getim