% trainPoseModel.m
% Script to train HRNet for Porcine Object Keypoint Detection.

% 1. Preprocess Data (Optional: Only if crops don't exist)
% preprocessPoseData(); 

% 2. Setup Datastores
[dsTrain, dsTest] = setupPoseDatastores();

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
options = trainingOptions("adam", ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-3, ...
    'BatchNormalizationStatistics', 'moving', ... % CRITICAL REQUIREMENT
    'ResetInputNormalization', false, ... % Required for Mask R-CNN training
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
[img, gtKpts, gtBox] = read(dsTest);
[bboxes, scores, labels, keypoints] = detect(trainedDetector, img);

% Overlay results
figure;
imshow(img);
hold on;
if ~isempty(keypoints)
    % Plot predicted keypoints
    plot(keypoints{1}(:,1), keypoints{1}(:,2), 'r*', 'MarkerSize', 10);
    % Plot bounding box
    rectangle('Position', bboxes(1,:), 'EdgeColor', 'r', 'LineWidth', 2);
end
title('Prediction (Red) vs Ground Truth (Blue)');
% Plot GT keypoints in blue for comparison
plot(gtKpts(:,1), gtKpts(:,2), 'b+', 'MarkerSize', 10);
hold off;
