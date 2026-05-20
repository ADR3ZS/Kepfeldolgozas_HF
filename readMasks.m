function masks = readMasks(imagePath)
    % READMASKS Custom read function for mask imageDatastore.
    %   imagePath: Full path to the RGB image.
    %
    %   Returns an H-by-W-by-N logical array of binary masks.

    % Map image path to label path
    % Assuming structure: .../images/... and .../labels/...
    labelPath = strrep(imagePath, 'images', 'labels');
    labelPath = strrep(labelPath, '.jpg', '.txt');
    
    if ~isfile(labelPath)
        % If no label file, return empty mask or handle as needed
        % For Mask R-CNN training, we usually expect labels.
        % We'll return an empty logical array if file missing.
        imgInfo = imfinfo(imagePath);
        masks = logical(zeros(imgInfo.Height, imgInfo.Width, 0));
        return;
    end
    
    imgInfo = imfinfo(imagePath);
    imgSize = [imgInfo.Height, imgInfo.Width];
    
    [masks, ~, ~] = parseSegmentationLabels(labelPath, imgSize);
end
