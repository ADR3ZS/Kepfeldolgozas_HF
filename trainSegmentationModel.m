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
    'NumStrongestRegions', 400, ...
    'NumRegionsToSample', 64);

% 5. Save Final Model
save('trainedMaskRCNN_final.mat', 'trainedMaskRCNN_final');

% 6. Basic Evaluation (Visual Check)
fprintf('Training complete. Displaying a test sample prediction...\n');

% Read the single 1-by-4 cell array from the combined datastore
data = read(dsTest); 
img = data{1};

% Perform instance segmentation using the final model
[bboxes, scores, labels, masks] = segmentObjects(trainedMaskRCNN_final, img);

% Safely check if any objects were detected before drawing
if isempty(bboxes) || size(bboxes, 1) == 0
    fprintf('No objects were detected in this test image.\n');
    imshow(img); % Display the raw image
else
    fprintf('%d objects detected. Displaying annotations...\n', size(bboxes, 1));
    % Overlay instance masks on the image
    imOverlay = insertObjectMask(img, masks);
    imshow(imOverlay);
    % Draw bounding boxes and labels
    showShape("rectangle", bboxes, "Label", labels, "Color", "red");
end

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
