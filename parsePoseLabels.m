function [bboxes, keypoints] = parsePoseLabels(labelPath, imgSize)
    % PARSEPOSELABELS Parse YOLO-style pose annotations.
    %   labelPath: Path to the .txt pose label file.
    %   imgSize:   [height, width] of the image.
    %
    %   bboxes:    N-by-4 matrix of [x, y, w, h] absolute bounding boxes.
    %   keypoints: N-by-1 cell array, each containing a 12-by-3 matrix [x, y, v].

    h = imgSize(1);
    w = imgSize(2);
    
    fid = fopen(labelPath, 'r');
    if fid == -1
        error('Could not open label file: %s', labelPath);
    end
    
    bboxes = [];
    keypoints = {};
    
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
        
        % YOLO Pose format: class, x_center, y_center, width, height, k1_x, k1_y, k1_v, ...
        % Box (relative to absolute)
        xc = data(2) * w;
        yc = data(3) * h;
        bw = data(4) * w;
        bh = data(5) * h;
        x1 = xc - bw/2;
        y1 = yc - bh/2;
        bboxes = [bboxes; [x1, y1, bw, bh]];
        
        % Keypoints (relative to absolute)
        kptsData = data(6:end);
        numKpts = 12;
        if numel(kptsData) >= numKpts * 3
            kptsMatrix = zeros(numKpts, 3);
            for k = 1:numKpts
                kx = kptsData(3*(k-1)+1) * w;
                ky = kptsData(3*(k-1)+2) * h;
                kv = kptsData(3*(k-1)+3);
                kptsMatrix(k, :) = [kx, ky, kv];
            end
            keypoints{end+1} = kptsMatrix; %#ok<AGROW>
        else
            keypoints{end+1} = zeros(numKpts, 3); %#ok<AGROW>
        end
        
        line = fgetl(fid);
    end
    fclose(fid);
end
