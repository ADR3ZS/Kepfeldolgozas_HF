function [dsTrain, dsTest] = setupPoseDatastores()
    % SETUPPOSEDATASTORES Prepare training and testing datastores for HRNet.

    imgDir = fullfile('data', 'pose_cropped', 'images');
    labelDir = fullfile('data', 'pose_cropped', 'labels');
    
    if ~exist(imgDir, 'dir') || ~exist(labelDir, 'dir')
        error('Cropped data not found. Run preprocessPoseData() first.');
    end
    
    imds = imageDatastore(imgDir, 'FileExtensions', '.jpg');
    numFiles = numel(imds.Files);
    
    if numFiles == 0
        error('No images found in %s', imgDir);
    end
    
    allKpts = cell(numFiles, 1);
    allBoxes = cell(numFiles, 1);
    allLabels = cell(numFiles, 1);
    
    fprintf('Loading %d cropped label files...\n', numFiles);
    for i = 1:numFiles
        [~, name, ~] = fileparts(imds.Files{i});
        labelPath = fullfile(labelDir, [name, '.mat']);
        
        if isfile(labelPath)
            data = load(labelPath);
            allKpts{i} = data.newKpts;
            allBoxes{i} = data.newBbox;
            allLabels{i} = categorical("pig");
        else
            error('Label file missing for image: %s', imds.Files{i});
        end
    end
    
    % Create individual datastores
    kptds = arrayDatastore(allKpts, 'IterationParameter', 'ByRecord');
    blds = boxLabelDatastore(table(allBoxes, allLabels));
    
    % Combine into {ImageData, Keypoint, BoundingBox}
    ds = combine(imds, kptds, blds);
    
    % Split 80% Train, 20% Test
    rng(0);
    shuffledIdx = randperm(numFiles);
    numTrain = floor(0.8 * numFiles);
    
    dsTrain = subset(ds, shuffledIdx(1:numTrain));
    dsTest = subset(ds, shuffledIdx(numTrain+1:end));
    
    fprintf('Pose data splitting complete: %d train, %d test samples.\n', numTrain, numFiles - numTrain);
end
