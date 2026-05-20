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
    'MaxEpochs', 1, ... % Restored for production training
    'MiniBatchSize', miniBatchSize, ... % Small batch size for Mask R-CNN
    'InitialLearnRate', 1e-5, ...
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
    'NumStrongestRegions', 200, ...
    'NumRegionsToSample', 32);

% 5. Save Final Model
save('trainedMaskRCNN_final.mat', 'trainedMaskRCNN_final');

% 6. Basic Evaluation (Visual Check)
fprintf('Training complete. Displaying a test sample prediction...\n');

% Read the single 1-by-4 cell array from the combined datastore
data = read(dsTest); 
img = data{1};

% Perform instance segmentation (CORRECTED OUTPUT ORDER)
[masks, labels, scores, bboxes] = segmentObjects(trainedMaskRCNN_final, img);

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

% 7. Quantitative Evaluation (Entire Test Set)
fprintf('\nEvaluating model on all test images. This may take a few minutes...\n');

% Reset the test datastore to start from the first image
reset(dsTest);

% The dataset was previously split into 20 test samples
numTestImages = 20;

% Initialize a table to collect the predictions as required by the evaluator
results = table('Size', [numTestImages, 4], ...
    'VariableTypes', {'cell', 'cell', 'cell', 'cell'}, ...
    'VariableNames', {'Masks', 'Labels', 'Scores', 'Boxes'});

% Run the network on each test image and collect outputs
for i = 1:numTestImages
    if hasdata(dsTest)
        data = read(dsTest);
        img = data{1};
        
        % Perform instance segmentation
        [masks, labels, scores, bboxes] = segmentObjects(trainedMaskRCNN_final, img);
        
        % Store the predictions in the table
        results.Masks{i} = masks;
        results.Labels{i} = labels;
        results.Scores{i} = scores;
        results.Boxes{i} = bboxes;
    end
end

% Reset the test datastore again so it can be used as the ground truth input
reset(dsTest);

% Evaluate the instance segmentation results against the ground truth
metrics = evaluateInstanceSegmentation(results, dsTest);

% Display the summary metrics in the Command Window
fprintf('\n--- Final Evaluation Metrics ---\n');
disp('Dataset Metrics:');
disp(metrics.DataSetMetrics);

disp('Class Metrics:');
disp(metrics.ClassMetrics);


