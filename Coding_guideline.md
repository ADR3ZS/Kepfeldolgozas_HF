# MATLAB Porcine Vision Model: Coding Guidelines & Handbook

## 1. Project Requirements and Toolboxes

To successfully utilize Mask R-CNN and HRNet algorithms in MATLAB, the following toolboxes and add-ons are mandatory:

- **Deep Learning Toolbox** [4, 5]
- **Computer Vision Toolbox** [5, 6]
- **Computer Vision Toolbox Model for Object Keypoint Detection** (Required to download pretrained HRNets and the HRNet object detector) [4, 7]

## 2. Model 1: Mask R-CNN for Instance Segmentation

Instance segmentation generates a pixel-level segmentation map for distinct objects [8]. You will use the `trainMaskRCNN` function.

### 2.1 Data Preparation for Mask R-CNN

Mask R-CNN networks require data to be provided in a specific 1-by-4 cell array format: `{RGB images, bounding boxes, labels, masks}` [9].

**Creating the Datastore:**
You must read the raw `.txt` polygon annotations and convert them into binary masks [10, 11].

1.  **Images:** Create an `imageDatastore` for RGB images [9].
2.  **Bounding Boxes & Labels:** Create a `boxLabelDatastore` [12].
3.  **Masks:** Create a custom read function combined with an `imageDatastore` to generate binary masks using `poly2mask`. The `poly2mask` function sets pixels inside the polygon to `1` and outside to `0` [10].
4.  **Combine:** Use the `combine` function to merge these datastores [12].

_Example: Converting Polygons to Binary Masks_

```matlab
% Assuming masks_polygon is a NumObjects-by-2 cell array of X,Y coordinates
denseMasks = false([h, w, numObjects]);
for i = 1:numObjects
    denseMasks(:,:,i) = poly2mask(masks_polygon{i}(:,1), masks_polygon{i}(:,2), h, w);
end
2.2 Designing and Training the Mask R-CNN
Initialize the Network: Configure the maskrcnn object, specifying class names (e.g., 'pig', 'partial pig') and anchor boxes
.
Train: Pass the combined datastore and the maskrcnn object to the trainMaskRCNN function
.
Example: Mask R-CNN Training Flow
% After combining datastores into 'trainingData'
options = trainingOptions("adam", 'MaxEpochs', 10, 'MiniBatchSize', 2);
trainedMaskRCNN = trainMaskRCNN(trainingData, maskrcnnObj, options);
3. Model 2: HRNet for Object Keypoint Detection
The High-Resolution Network (HRNet) maintains high-resolution representations throughout the network to accurately localize keypoints
.
3.1 Data Preparation for HRNet
HRNet requires a datastore that returns a 1-by-3 cell array: {ImageData, Keypoint, BoundingBox}
.
Creating the Datastore:
Images: Because keypoint detection requires cropped images with only a single annotation, crop the image around the bounding box and save the new crops
. Read them using an imageDatastore
.
Keypoints: Create an arrayDatastore for ground truth keypoints
. Keypoints must be formatted as an N-by-3 matrix per object: [x, y, v] where v is visibility (1 for occluded, 2 for visible)
.
Bounding Boxes: Create a boxLabelDatastore for the bounding boxes
. Bounding boxes must be in [x, y, w, h] absolute pixel formats
.
Combine: Use the combine function to merge the three datastores
.
3.2 Designing and Training HRNet
You can perform transfer learning using a pretrained HRNet-W32 or HRNet-W48 model
.
Define Classes: Define the specific keypoints for your object (e.g., Nose, Head, Tail)
.
Initialize the Network: Use hrnetObjectKeypointDetector("human-full-body-w32", keypointClasses) to configure the base network for custom transfer learning
.
Train: Train using trainHRNetObjectKeypointDetector. Note: The training options MUST set BatchNormalizationStatistics to "moving"
.
Example: HRNet Training Flow
% Define classes
keypointClasses = ["Nose", "Head", "LShoulder", "RShoulder", "LPaw", "RPaw", "Torso", "LHip", "RHip", "Tail", "LFoot", "RFoot"]';

% Initialize detector
keypointDetector = hrnetObjectKeypointDetector("human-full-body-w32", keypointClasses);

% Define training options
options = trainingOptions("adam", ...
    MaxEpochs=20, ...
    MiniBatchSize=16, ...
    BatchNormalizationStatistics="moving"); % Mandatory requirement

% Train
trainedKeypointDetector = trainHRNetObjectKeypointDetector(trainingData, keypointDetector, options);

***
```
