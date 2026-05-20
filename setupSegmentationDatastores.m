function [dsTrain, dsTest] = setupSegmentationDatastores()
    % SETUPSEGMENTATIONDATASTORES Prepare training and testing datastores for Mask R-CNN.

    % 1. Find images
    imgDir = fullfile('data', 'segmentation', 'images', 'train');
    imds = imageDatastore(imgDir, 'IncludeSubfolders', true, 'FileExtensions', '.jpg');

    % 2. Map to labels and filter
    imageFiles = imds.Files;
    % Assuming parallel structure: images/train/... and labels/train/...
    labelFiles = strrep(imageFiles, 'images', 'labels');
    labelFiles = strrep(labelFiles, '.jpg', '.txt');

    validIdx = false(size(labelFiles));
    for i = 1:numel(labelFiles)
        if isfile(labelFiles{i})
            validIdx(i) = true;
        end
    end
    
    if ~any(validIdx)
        error('No valid label files found in %s', fullfile('data', 'segmentation', 'labels', 'train'));
    end
    
    imds.Files = imageFiles(validIdx);
    labelFiles = labelFiles(validIdx);

    % 3. Pre-parse labels for boxLabelDatastore
    numFiles = numel(imds.Files);
    allBoxes = cell(numFiles, 1);
    allLabels = cell(numFiles, 1);

    fprintf('Parsing %d label files...\n', numFiles);
    for i = 1:numFiles
        imgInfo = imfinfo(imds.Files{i});
        [~, bboxes, labels] = parseSegmentationLabels(labelFiles{i}, [imgInfo.Height, imgInfo.Width]);
        allBoxes{i} = bboxes;
        allLabels{i} = labels;
    end

    blds = boxLabelDatastore(table(allBoxes, allLabels));

    % 4. Custom Mask Datastore
    maskds = imageDatastore(imds.Files, 'ReadFcn', @readMasks);

    % 5. Data Splitting (80% Train, 20% Test)
    rng(0); % For reproducibility
    shuffledIdx = randperm(numFiles);
    numTrain = floor(0.8 * numFiles);

    trainIdx = shuffledIdx(1:numTrain);
    testIdx = shuffledIdx(numTrain+1:end);

    % Training set
    imdsTrain = subset(imds, trainIdx);
    bldsTrain = subset(blds, trainIdx);
    maskdsTrain = subset(maskds, trainIdx);
    dsTrain = combine(imdsTrain, bldsTrain, maskdsTrain);

    % Testing set
    imdsTest = subset(imds, testIdx);
    bldsTest = subset(blds, testIdx);
    maskdsTest = subset(maskds, testIdx);
    dsTest = combine(imdsTest, bldsTest, maskdsTest);
    
    fprintf('Data splitting complete: %d train, %d test samples.\n', numTrain, numFiles - numTrain);
end
