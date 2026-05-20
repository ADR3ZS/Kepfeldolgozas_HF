function preprocessPoseData()
    % PREPROCESSPOSEDATA Crop images around bounding boxes and transform coordinates.
    % Saves results to data/pose_cropped/

    sourceImgDir = fullfile('data', 'pose', 'images', 'train');
    sourceLabelDir = fullfile('data', 'pose', 'labels', 'train');
    
    outputImgDir = fullfile('data', 'pose_cropped', 'images');
    outputLabelDir = fullfile('data', 'pose_cropped', 'labels');
    
    if ~exist(outputImgDir, 'dir'), mkdir(outputImgDir); end
    if ~exist(outputLabelDir, 'dir'), mkdir(outputLabelDir); end
    
    imds = imageDatastore(sourceImgDir, 'IncludeSubfolders', true, 'FileExtensions', '.jpg');
    numFiles = numel(imds.Files);
    
    fprintf('Preprocessing %d images for pose detection...\n', numFiles);
    
    cropCount = 0;
    for i = 1:numFiles
        imagePath = imds.Files{i};
        % Map to label path
        labelPath = strrep(imagePath, 'images', 'labels');
        labelPath = strrep(labelPath, '.jpg', '.txt');
        
        if ~isfile(labelPath), continue; end
        
        img = imread(imagePath);
        [h, w, ~] = size(img);
        
        [bboxes, allKeypoints] = parsePoseLabels(labelPath, [h, w]);
        
        numObjects = size(bboxes, 1);
        for j = 1:numObjects
            cropCount = cropCount + 1;
            
            bbox = bboxes(j, :);
            kpts = allKeypoints{j};
            
            % 1. Crop Image
            % Ensure bbox is within image bounds
            x1 = max(1, floor(bbox(1)));
            y1 = max(1, floor(bbox(2)));
            x2 = min(w, ceil(bbox(1) + bbox(3)));
            y2 = min(h, ceil(bbox(2) + bbox(4)));
            
            if x2 <= x1 || y2 <= y1, continue; end
            
            imgCrop = img(y1:y2, x1:x2, :);
            
            % 2. Recalculate Coordinates
            % New bbox in crop space
            newBbox = [bbox(1)-x1+1, bbox(2)-y1+1, bbox(3), bbox(4)];
            
            % New keypoints in crop space
            newKpts = kpts;
            newKpts(:, 1) = kpts(:, 1) - x1 + 1;
            newKpts(:, 2) = kpts(:, 2) - y1 + 1;
            
            % 3. Save Cropped Image
            cropName = sprintf('pig_crop_%05d.jpg', cropCount);
            imwrite(imgCrop, fullfile(outputImgDir, cropName));
            
            % 4. Save New Labels (.mat format for easy MATLAB loading later)
            labelName = sprintf('pig_crop_%05d.mat', cropCount);
            save(fullfile(outputLabelDir, labelName), 'newBbox', 'newKpts');
        end
    end
    
    fprintf('Preprocessing complete. Created %d crops.\n', cropCount);
end
