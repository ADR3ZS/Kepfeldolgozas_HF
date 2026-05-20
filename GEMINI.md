# Porcine Vision Model Training (MATLAB)

This project aims to implement a vision system for porcine (pig) analysis using MATLAB. The system focuses on two primary tasks: Instance Segmentation for identifying pigs and their parts, and Keypoint Detection for anatomical analysis.

## Project Overview

- **Task 1: Instance Segmentation (Mask R-CNN)**
  - Goal: Segment pigs and partial pigs in images.
  - Classes: `0` (pig), `6` (partial pig).
  - Data: Polygon annotations in YOLO-style `.txt` files.
- **Task 2: Keypoint Detection (HRNet)**
  - Goal: Identify 12 anatomical keypoints on pigs.
  - Keypoints: Nose, Head, LShoulder, RShoulder, LPaw, RPaw, Torso, LHip, RHip, Tail, LFoot, RFoot.
  - Data: Pose annotations in YOLO-style `.txt` files, subsequently cropped for single-instance training.

## Tech Stack

- **MATLAB**
- **Deep Learning Toolbox**
- **Computer Vision Toolbox**
- **Computer Vision Toolbox Model for Object Keypoint Detection** (Add-on)

## Directory Structure

- `data/pose/`: Raw images and YOLO-style pose annotations.
- `data/pose_cropped/`: Images cropped around individual pigs with transformed keypoint coordinates (for HRNet training).
- `data/segmentation/`: Raw images and YOLO-style polygon annotations.
- `Project_task_prompt.md`: Detailed requirements and step-by-step instructions.
- `Coding_guideline.md`: MATLAB-specific implementation details and toolbox requirements.

## Development Workflow

### Task 1: Mask R-CNN Pipeline
1. **Parse Annotations:** Convert relative YOLO coordinates to absolute pixel values.
2. **Create Masks:** Generate binary masks from polygons using `poly2mask`.
3. **Datastore Construction:** Combine `imageDatastore`, `boxLabelDatastore`, and a custom mask `imageDatastore` into a `{RGB, Box, Label, Mask}` format.
4. **Training:** Initialize `maskrcnn` and train using `trainMaskRCNN`.

### Task 2: HRNet Pipeline
1. **Parse Pose:** Read bounding box and keypoint (triplet: X, Y, Visibility) data.
2. **Preprocessing (CRITICAL):**
   - Crop images around each pig instance.
   - Recalculate coordinates for the cropped space.
3. **Datastore Construction:** Combine into `{ImageData, Keypoint, BoundingBox}` format.
4. **Training:** Use `hrnetObjectKeypointDetector` with transfer learning (`human-full-body-w32`).
   - **Requirement:** `BatchNormalizationStatistics` must be set to `"moving"` in `trainingOptions`.

## Key Commands (MATLAB)

*Note: Scripts are to be implemented based on `Project_task_prompt.md`.*

- **TODO:** Implement `train_segmentation.m` for Task 1.
- **TODO:** Implement `train_keypoints.m` for Task 2.

## Coding Conventions

- **Datastores:** Strictly adhere to the cell array formats required by `trainMaskRCNN` and `trainHRNetObjectKeypointDetector`.
- **Pre-trained Models:** Use `human-full-body-w32` as the base for HRNet transfer learning.
- **Coordinates:** Always ensure coordinates are denormalized and transformed correctly when cropping.
