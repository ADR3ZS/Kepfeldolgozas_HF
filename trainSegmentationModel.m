% trainSegmentationModel.m
% Script to train Mask R-CNN for Porcine Instance Segmentation.

% 1. Setup Datastores
[dsTrain, dsTest] = setupSegmentationDatastores();

% 2. Model Configuration
classNames = ["pig", "partial pig"];

% Define anchor boxes (standard Mask R-CNN defaults or tuned)
anchorBoxes = [
    32 32; 64 64; 128 128; 256 256; 512 512; % Squares
    32 64; 64 32; 64 128; 128 64; 128 256; 256 128 % Rectangles
];

% Initialize Mask R-CNN object
% Using ResNet-50 as a common backbone if not specified
maskrcnnObj = maskrcnn("resnet50-coco", classNames, anchorBoxes);

% 3. Training Options
miniBatchSize = 1;
numTrainingSamples = numel(dsTrain.UnderlyingDatastores{1}.Files);
iterationsPerEpoch = floor(numTrainingSamples / miniBatchSize);
drasticValFreq = iterationsPerEpoch * 5; 

options = trainingOptions("adam", ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', miniBatchSize, ... % Small batch size for Mask R-CNN
    'InitialLearnRate', 1e-4, ...
    'ResetInputNormalization', false, ... % Required for Mask R-CNN training
    'ValidationData', dsTest, ...
    'ValidationFrequency', drasticValFreq, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% 4. Train Model
fprintf('Starting Mask R-CNN training...\n');
trainedMaskRCNN = trainMaskRCNN(dsTrain, maskrcnnObj, options, ...
    'NumStrongestRegions', 100, ...
    'NumRegionsToSample', 16);

% 5. Save Model
save('trainedMaskRCNN.mat', 'trainedMaskRCNN');

% 6. Basic Evaluation (Visual Check)
fprintf('Training complete. Displaying a test sample prediction...\n');
[img, gtBoxes, gtLabels, gtMasks] = read(dsTest);
[bboxes, scores, labels, masks] = segmentObjects(trainedMaskRCNN, img);

% Overlay results
figure;
subplot(1,2,1);
imshow(img);
showShape("rectangle", gtBoxes, 'Label', gtLabels);
title('Ground Truth');

subplot(1,2,2);
imshow(img);
if ~isempty(bboxes)
    showShape("rectangle", bboxes, 'Label', labels);
    % Note: overlaying masks requires additional logic or insertObjectMask
end
title('Prediction');
