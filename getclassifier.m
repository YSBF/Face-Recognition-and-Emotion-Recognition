outputFolder = fullfile('D:\AI\imlib-e');
rootFolder = fullfile(outputFolder, 'Photo');
imgSets = [imageSet(fullfile(rootFolder, 'happy')), ...
    imageSet(fullfile(rootFolder, 'plain')), ...
    imageSet(fullfile(rootFolder, 'sad')),imageSet(fullfile(rootFolder,'surprised'))];
minSetCount = min([imgSets.Count]);
imgSets = partition(imgSets, minSetCount, 'randomize');
[trainingSets, validationSets] = partition(imgSets, 0.8, 'randomize');
bag = bagOfFeatures(trainingSets);
eCategoryClassifier = trainImageCategoryClassifier(trainingSets, bag);
outputFolder = fullfile('D:\AI\imagelib - ¸±±¾');
rootFolder = fullfile(outputFolder, 'Photo');
imgSets = [imageSet(fullfile(rootFolder, 'gmh')), ...
    imageSet(fullfile(rootFolder, 'Xiaoyi'))];
minSetCount = min([imgSets.Count]);
imgSets = partition(imgSets, minSetCount, 'randomize');
[trainingSets, validationSets] = partition(imgSets, 0.9, 'randomize');
bag = bagOfFeatures(trainingSets);
categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);