function data = downsampleData(data, targetSize)
    % DOWNSAMPLEDATA Resizes image, bounding boxes, and masks for Mask R-CNN.
    % data is a 1-by-4 cell array: {image, bbox, label, mask}
    img = data{1};
    bboxes = data{2};
    labels = data{3};
    masks = data{4};
    
    % 1. Resize Image safely
    originalSize = [size(img, 1), size(img, 2)];
    imgResized = imresize(img, targetSize);
    
    % 2. Resize Bounding Boxes
    scale = targetSize ./ originalSize;
    bboxesResized = bboxresize(bboxes, scale);
    
    % 3. Resize Masks (must use 'nearest' to remain binary)
    masksResized = imresize(masks, targetSize, "nearest");
    
    % Return the updated 1-by-4 cell array
    data = {imgResized, bboxesResized, labels, masksResized};
end
