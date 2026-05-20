```markdown
# AI Agent Instruction Prompt: Porcine Vision Model Training in MATLAB

**Context:**
You are a senior MATLAB AI engineer. Your task is to implement a vision system involving Instance Segmentation and Keypoint Detection based on a custom dataset of pigs. The dataset contains `images/` (.jpg) and `labels/` (.txt) [28, 29].
You must strictly follow these instructions to script the data preparation and model training pipelines.

## Task 1: Train Instance Segmentation Model (Mask R-CNN)

**Goal:** Train a Mask R-CNN model to segment pigs and partial pigs.

**Step-by-Step Instructions:**

1.  **Parse Annotations:** Write a custom parser to read the `.txt` label files.
    - _Format:_ Each line represents a bounding box/polygon. First value is the class ID. `0` = pig, `6` = partial pig. Subsequent values are X and Y relative coordinates (0-1) [29].
    - _Action:_ Denormalize the X and Y coordinates by multiplying them by the corresponding image width and height.
2.  **Create Binary Masks:** Use the `poly2mask` function to convert the sequence of X,Y polygon coordinates into a binary mask for each instance [10, 11].
3.  **Construct Datastores:**
    - Create an `imageDatastore` for the raw `.jpg` files.
    - Create a `boxLabelDatastore` for the bounding boxes and labels.
    - Create an `imageDatastore` with a custom read function that outputs the generated binary masks [9, 12].
    - Combine them so the output format is exactly `{RGB images, bounding boxes, labels, masks}` [9].
4.  **Data Splitting:** Randomly split the combined datastore into 80% training data and 20% testing data [18].
5.  **Model Configuration & Training:** Initialize a `maskrcnn` object utilizing the 2 target classes. Configure `trainingOptions` and train the network utilizing the `trainMaskRCNN` function [13, 14].

## Task 2: Train Keypoint Detection Model (HRNet)

**Goal:** Train an HRNet-based keypoint detector (`hrnetObjectKeypointDetector`) to identify 12 anatomical keypoints on pigs [28].

**Step-by-Step Instructions:**

1.  **Parse Pose Annotations:** Read the `.txt` pose label files.
    - _Format Check:_ Class ID is `0`. The next 4 values are bounding box features: `X_center`, `Y_center`, `width`, `height` (relative 0-1) [20].
    - _Keypoint Check:_ The remaining values are triplets: `X_relative`, `Y_relative`, `Visibility` (`1` = occluded, `2` = visible) [20].
    - _Order Check:_ Nose, Head, LShoulder, RShoulder, LPaw, RPaw, Torso, LHip, RHip, Tail, LFoot, RFoot [18].
2.  **Image Cropping & Coordinate Transformation (CRITICAL):**
    - HRNet requires images with only _one_ annotation [18]. Iterate through the dataset and crop the raw image around the calculated bounding box. Save these cropped images to a new directory [18].
    - Recalculate the absolute X and Y coordinates of both the bounding boxes and the keypoints so they reflect the new cropped image space.
3.  **Construct Datastores:**
    - Create an `imageDatastore` pointing to the _newly cropped_ images [19].
    - Create an `arrayDatastore` containing the N-by-3 keypoint matrices `[x, y, visibility]` [19, 21].
    - Create a `boxLabelDatastore` for the bounding boxes `[x, y, w, h]` [19, 22].
    - Combine them into a single datastore outputting exactly `{ImageData, Keypoint, BoundingBox}` [17].
4.  **Data Splitting:** Split the datastore into 80% training and 20% testing [18].
5.  **Model Configuration & Training:**
    - Instantiate the detector using transfer learning: `hrnetObjectKeypointDetector("human-full-body-w32", keypointClasses)` where `keypointClasses` is an array of the 12 string labels mentioned above [23-25].
    - Configure `trainingOptions`. _Requirement:_ You MUST set the `BatchNormalizationStatistics` parameter to `"moving"` [26, 27].
    - Train the model using `trainHRNetObjectKeypointDetector` [2, 26].
```
