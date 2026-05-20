% trainSegmentationModel.m
% Script to train Mask R-CNN for Porcine Instance Segmentation.

% 1. Setup Datastores
[dsTrain, dsTest, numTrainingSamples] = setupSegmentationDatastores();

% 1.1 Downsample Data
targetSize = [224, 224];
dsTrain = transform(dsTrain, @(data)downsampleData(data, targetSize));
dsTest = transform(dsTest, @(data)downsampleData(data, targetSize));

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
iterationsPerEpoch = floor(numTrainingSamples / miniBatchSize);
drasticValFreq = iterationsPerEpoch * 1; 

options = trainingOptions("adam", ...
    'MaxEpochs', 2, ... % Restored for production training
    'MiniBatchSize', miniBatchSize, ... % Small batch size for Mask R-CNN
    'InitialLearnRate', 1e-4, ...
    'ResetInputNormalization', false, ... % Required for Mask R-CNN training
    'ValidationData', dsTest, ...
    'ValidationFrequency', drasticValFreq, ...
    'ExecutionEnvironment', 'gpu', ... % Force GPU execution
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% 4. Resume Training from the Existing Model
fprintf('Resuming Mask R-CNN training for production...\n');

% Ensure the previous model is loaded if not in workspace
if ~exist('trainedMaskRCNN', 'var') && isfile('trainedMaskRCNN.mat')
    load('trainedMaskRCNN.mat', 'trainedMaskRCNN');
end

% Resume training using the existing model and default region proposal values
trainedMaskRCNN_final = trainMaskRCNN(dsTrain, trainedMaskRCNN, options, ...
    'NumStrongestRegions', 1000, ...
    'NumRegionsToSample', 128);

% 5. Save Final Model
save('trainedMaskRCNN_final.mat', 'trainedMaskRCNN_final');

% 6. Basic Evaluation (Visual Check)
fprintf('Training complete. Displaying a test sample prediction...\n');

% Read the single 1-by-4 cell array from the combined datastore
data = read(dsTest); 

% Unpack the cell array into individual variables
img = data{1};
gtBoxes = data{2};
gtLabels = data{3};
gtMasks = data{4};

% Perform instance segmentation using the final model
[bboxes, scores, labels, masks] = segmentObjects(trainedMaskRCNN_final, img);

% Overlay results
figure;

subplot(1,2,1);
if ~isempty(gtBoxes)
    imshow(img);
    hold on;
    showShape("rectangle", gtBoxes, 'Label', gtLabels);
    hold off;
else
    fprintf('No ground truth objects in this test image.\n');
    imshow(img);
end
title('Ground Truth');

subplot(1,2,2);
if ~isempty(bboxes)
    imshow(img);
    hold on;
    showShape("rectangle", bboxes, 'Label', labels);
    % Note: overlaying masks requires additional logic or insertObjectMask
    hold off;
else
    fprintf('No objects were detected in this test image.\n');
    imshow(img); % Just show the raw image
end
title('Prediction');

function data = downsampleData(data, targetSize)
    % data is a 1-by-4 cell array: {image, bbox, label, mask}
    img = data{1};
    bboxes = data{2};
    labels = data{3};
    masks = data{4};
    
    % 1. Resize Image
    originalSize = size(img, [1, 2]);
    imgResized = imresize(img, targetSize);
    
    % 2. Resize Bounding Boxes
    scale = targetSize ./ originalSize;
    bboxesResized = bboxresize(bboxes, scale);
    
    % 3. Resize Masks (must use 'nearest' to remain binary)
    masksResized = imresize(masks, targetSize, "nearest");
    
    % Return the updated 1-by-4 cell array
    data = {imgResized, bboxesResized, labels, masksResized};
end
