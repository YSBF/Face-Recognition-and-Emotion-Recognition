faceDetector = vision.CascadeObjectDetector();
clear cam
cam = webcam();
videoFrame = snapshot(cam);
frameSize = size(videoFrame);
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);
runLoop = true;
numPts = 0;
frameCount = 0;
n=1;
while runLoop
    videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;
    bbox = faceDetector.step(videoFrameGray);
    if ~isempty(bbox)
        points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));
        bboxPoints = bbox2points(bbox(1, :));
        selectVideoFrame=imcrop(videoFrame,[min(bboxPoints(:,1)),min(bboxPoints(:,2)),max(bboxPoints(:,1))-min(bboxPoints(:,1)),max(bboxPoints(:,2))-min(bboxPoints(:,2))]);
        selectVideoFrame=imresize(selectVideoFrame,[300,300]);
        imshow(rgb2gray(selectVideoFrame))
        img = readAndPreprocessImage(selectVideoFrame);
        [labelIdx, scores] = predict(categoryClassifier, img);
        [elabelIdx, escores] = predict(eCategoryClassifier, img);
        categoryClassifier.Labels(labelIdx)
        eCategoryClassifier.Labels(elabelIdx)
        fprintf('emotion scores: happy plain sad surprised\n')
        fprintf(num2str(escores))
        fprintf('\n')
        fprintf('Person Scores: gmh Xiaoyi\n')
        fprintf(num2str(scores))
        fprintf('\n')
        videoFrame=rgb2gray(videoFrame);
        bboxPolygon = reshape(bboxPoints', 1, []);
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
        videoFrame = insertText(videoFrame,[1,1],num2str(frameCount));
        if(min(scores)<-0.6)
            videoFrame = insertText(videoFrame,[1,21],categoryClassifier.Labels(labelIdx));
        end
        if(min(escores)<-0.4)
            videoFrame = insertText(videoFrame,[1,41],eCategoryClassifier.Labels(elabelIdx));
        end
        
    end
    step(videoPlayer,videoFrame);
    runLoop = isOpen(videoPlayer);
end
clear cam;
release(videoPlayer);
release(faceDetector);