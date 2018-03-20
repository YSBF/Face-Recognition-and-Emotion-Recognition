fprintf('Test for Gou Minghao\n')
for n=1:245
    newImage = fullfile(rootFolder, 'gmh', strcat('a (',num2str(n),').jpg'));
    % Pre-process the images as required for the CNN
    img = readAndPreprocessImagef(newImage);
    % Extract image features using the CNN
    imageFeatures = activations(convnet, img, featureLayer);

    fprintf('%d',n)
    fprintf(':')
    % Make a prediction using the classifier
    label = predict(classifier, imageFeatures);
    switch(label)
        case 'gmh'
            fprintf('gmh')
        case 'Xiaoyi'
            fprintf('Xiaoyi   WRONG!')
    end
    fprintf('\n')
end


fprintf('Test for Gou Minghao\n')
for n=1:3000
    newImage = fullfile(rootFolder, 'Xiaoyi', strcat('a (',num2str(n),').jpg'));
    % Pre-process the images as required for the CNN
    img = readAndPreprocessImagef(newImage);
    % Extract image features using the CNN
    imageFeatures = activations(convnet, img, featureLayer);

    fprintf('%d',n)
    fprintf(':')
    % Make a prediction using the classifier
    label = predict(classifier, imageFeatures);
    switch(label)
        case 'gmh'
            fprintf('gmh WRONG!')
        case 'Xiaoyi'
            fprintf('Xiaoyi')
    end
    fprintf('\n')
end

