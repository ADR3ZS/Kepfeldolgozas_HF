% trainPoseModel.m
% Script to train HRNet for Porcine Object Keypoint Detection.

% 1. Preprocess Data (Optional: Only if crops don't exist)
% preprocessPoseData(); 

% 2. Setup Datastores
[dsTrain, dsTest, numTrainingSamples] = setupPoseDatastores();

% 2.1 Implement Data Downsampling (Custom Transform)
targetSize = [224, 224];
dsTrain = transform(dsTrain, @(data)downsamplePoseData(data, targetSize));
dsTest = transform(dsTest, @(data)downsamplePoseData(data, targetSize));

% 3. Model Configuration
keypointClasses = [ ...
    "Nose", "Head", "LShoulder", "RShoulder", ...
    "LPaw", "RPaw", "Torso", "LHip", "RHip", ...
    "Tail", "LFoot", "RFoot" ...
]';

% Instantiate detector using transfer learning
fprintf('Initializing HRNet detector...\n');
detector = hrnetObjectKeypointDetector("human-full-body-w32", keypointClasses);

% 4. Training Options
miniBatchSize = 1;
iterationsPerEpoch = floor(numTrainingSamples / miniBatchSize);

options = trainingOptions("adam", ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', 1e-5, ... % Lower learning rate for stable transfer learning
    'GradientThreshold', 1, ...
    'GradientThresholdMethod', 'l2norm', ...
    'BatchNormalizationStatistics', 'moving', ... % CRITICAL REQUIREMENT
    'ResetInputNormalization', false, ... 
    'ValidationData', dsTest, ...
    'ValidationFrequency', iterationsPerEpoch * 2, ... % Validate every 2 epochs
    'ExecutionEnvironment', 'auto', ... % Utilize GPU with auto fallback
    'PreprocessingEnvironment', 'background', ... % Prevent CPU from starving the GPU
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% 5. Train Model
fprintf('Starting HRNet training...\n');
trainedDetector = trainHRNetObjectKeypointDetector(dsTrain, detector, options);

% 6. Save Model
save('trainedPoseDetector.mat', 'trainedDetector');

% 7. Basic Evaluation (Visual Check)
fprintf('Training complete. Displaying a test sample prediction...\n');

% Fix CombinedDatastore Read Error
data = read(dsTest);
img = data{1};
gtKpts = data{2};
gtBox = data{3};

% Correct HRNet inference syntax
[keypoints, keypointScores] = detect(trainedDetector, img, gtBox);

% Visualize
figure;
if ~isempty(keypoints)
    % Insert keypoints and the bounding box
    imgOut = insertObjectKeypoints(img, keypoints, "KeypointColor", "red");
    imgOut = insertShape(imgOut, "rectangle", gtBox, "Color", "blue", "LineWidth", 3);
    imshow(imgOut);
    title('HRNet Pose Estimation Results');
else
    fprintf('No keypoints were detected in this test image.\n');
    imshow(img);
end

function data = downsamplePoseData(data, targetSize)
    % data is a 1-by-3 cell array: {image, keypoints, bbox}
    img = data{1}; 
    kpts = data{2}; 
    bbox = data{3};
    originalSize = size(img, [1, 2]);
    
    % 1. Resize Image
    imgResized = imresize(img, targetSize);
    
    % 2. Calculate scales
    scaleY = targetSize(1) / originalSize(1);
    scaleX = targetSize(2) / originalSize(2);
    
    % 3. Resize Bounding Box
    bboxResized = bboxresize(bbox, [scaleY, scaleX]);
    
    % 4. Resize Keypoints (Scale X and Y only)
    kptsResized = kpts;
    kptsResized(:,1) = kpts(:,1) * scaleX;
    kptsResized(:,2) = kpts(:,2) * scaleY;
    
    data = {imgResized, kptsResized, bboxResized};
end
