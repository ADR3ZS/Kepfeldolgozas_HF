function [masks, bboxes, labels] = parseSegmentationLabels(labelPath, imgSize)
    % PARSESEGMENTATIONLABELS Parse YOLO-style polygon annotations.
    %   labelPath: Path to the .txt label file.
    %   imgSize:   [height, width] of the image.
    %
    %   masks:  H-by-W-by-N logical array of binary masks.
    %   bboxes: N-by-4 matrix of [x, y, w, h] bounding boxes.
    %   labels: N-by-1 categorical array of labels.

    h = imgSize(1);
    w = imgSize(2);
    
    fid = fopen(labelPath, 'r');
    if fid == -1
        error('Could not open label file: %s', labelPath);
    end
    
    masks = logical([]);
    bboxes = [];
    labels = string([]);
    
    line = fgetl(fid);
    while ischar(line)
        if isempty(strtrim(line))
            line = fgetl(fid);
            continue;
        end
        
        data = str2num(line); %#ok<ST2NM>
        if isempty(data)
            line = fgetl(fid);
            continue;
        end
        
        classID = data(1);
        
        % Filter labels to only include 'pig' (0) and 'partial pig' (6)
        if classID == 0
            currentLabel = "pig";
        elseif classID == 6
            currentLabel = "partial pig";
        else
            line = fgetl(fid);
            continue;
        end
        
        polyCoords = data(2:end);
        
        % Denormalize
        x = polyCoords(1:2:end) * w;
        y = polyCoords(2:2:end) * h;
        
        % Ensure coordinates are within image bounds
        x = max(1, min(w, x));
        y = max(1, min(h, y));
        
        % Binary Mask
        if numel(x) >= 3 % Need at least 3 points for a polygon
            m = poly2mask(x, y, h, w);
            masks = cat(3, masks, m);
            
            % Bounding Box [x y w h]
            x1 = min(x);
            y1 = min(y);
            x2 = max(x);
            y2 = max(y);
            bboxes = [bboxes; [x1, y1, x2-x1, y2-y1]];
            
            labels = [labels; currentLabel];
        end
        
        line = fgetl(fid);
    end
    fclose(fid);
    
    % Convert labels to categorical
    if isempty(labels)
        labels = categorical([], ["pig", "partial pig"]);
    else
        labels = categorical(labels, ["pig", "partial pig"]);
    end
end
