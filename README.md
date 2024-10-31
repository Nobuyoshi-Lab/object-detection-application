# Hack Week Challenge: Object Detection Application

## Overview

This project focuses on object detection in images of varying resolutions and complexities using advanced models and techniques. It addresses challenges associated with detecting objects in high-resolution images and provides solutions to improve detection accuracy and visualization.

## Prerequisites

To run the application, ensure you have the following libraries installed:

```
pip install customtkinter emoji opencv-python numpy ultralytics sahi torchvision torch piexif Pillow
```

**Note:** For optimal performance with GPU acceleration, visit the [PyTorch website](https://pytorch.org/get-started/locally/) to choose the appropriate installation command based on your system.

## Issues Addressed

### High-Resolution Images with Varying Complexity

- **Image 1-8, 10-16, 18-20**
  - **Complexity**: Low (large, centralized objects)
  - **Slicing Aided Hyper Inference (SAHI)**: Not necessary due to low complexity and prominent objects.

- **Image 17 (Food Platter) & Image 9 (Beach Scene)**
  - **Complexity**: High (numerous small, scattered objects)
  - **Slicing Aided Hyper Inference (SAHI)**: Required to effectively detect small objects scattered throughout the image.

### Detection Accuracy

- **Underrepresented Classes**: Difficulty in accurately detecting certain objects (e.g., specific food items) due to insufficient representation in pre-trained models.
- **Small Object Detection**: Challenges in detecting small objects within large, complex images, often leading to missed detections.

### Bounding Box and Labeling Issues

- **Overlapping Detections**: Multiple bounding boxes overlap, causing cluttered visualizations and confusion.
- **Label Placement**: Labels sometimes exceed bounding box boundaries, especially when boxes are small or labels are lengthy.

## Approaches and Implemented Solutions

### Models Used

- **Large Images**: Utilized SAHI with **YOLO11** (`yolo11x.pt`) for slicing and detection.
- **Small Images**: Employed **YOLOv8** pre-trained on the Open Images V7 dataset (`yolov8x-oiv7.pt`).

### Handling High-Resolution Images with High Complexity

1. **Slicing with SAHI**
   - **Technique**: Divided large images into smaller tiles.
   - **Inference**: Performed detection on each tile using YOLO11.
   - **Aggregation**: Combined detections from all tiles for final results.

2. **Non-Maximum Suppression (NMS)**
   - **Purpose**: Reduced overlapping bounding boxes.
   - **Method**: Applied per-class NMS to handle overlapping detections effectively.

3. **Bounding Box Grouping**
   - **Process**: Used connected components analysis and morphological operations.
   - **Result**: Merged overlapping or adjacent bounding boxes of the same class.
   - **Labeling**: Adjusted labels to reflect grouped detections.

### Handling Low Complexity and Low-Resolution Images

1. **Image Resizing**
   - **Approach**: Scaled images to manageable sizes while preserving aspect ratios.
   - **Detection**: Used standard YOLOv8 detection without slicing.

2. **Leveraging Pre-trained Models**
   - **Model**: YOLOv8 trained on Open Images V7.
   - **Advantage**: Benefited from a wide range of classes and extensive training data.

### Enhancements in Labeling and Visualization

- **Dynamic Font Scaling**
  - Adjusted font sizes to ensure labels fit within bounding boxes.
  - Implemented scaling to accommodate various box sizes.

- **Optimized Label Positioning**
  - Placed labels within bounding boxes with proper padding.
  - Ensured labels are clearly visible and do not overflow.

## Challenges and Solutions

### Detecting Small Objects

- **Challenge**: Small objects are often missed or inaccurately detected.
- **Solution**: Implemented multi-scale testing and fine-tuned parameters to improve small object detection.

### Overlapping Objects and Bounding Boxes

- **Challenge**: Overlapping detections lead to cluttered images.
- **Solution**: Used advanced grouping and merging techniques, including connected components and morphological operations.

## Graphical User Interface (GUI) Features

- **User-Friendly Interface**: Provides an intuitive GUI for users to interact with the application.
- **Input Selection**: Allows users to select the input image directory.
- **Output Viewing**: Enables users to open the output folder containing processed images.
- **Adjustable Parameters**: Users can adjust confidence thresholds, NMS IoU thresholds, and image size thresholds through sliders and input fields.
- **SAHI Toggle**: Provides a checkbox to enable or disable SAHI for large images.
- **Processing Control**: Features a "Start Processing" button that turns into a "Cancel" button during processing, allowing users to stop the process at any time.
- **Progress Monitoring**: Displays real-time status updates and a progress bar during image processing.
- **Theme Customization**: Offers options to change the appearance of the GUI (System, Light, Dark themes).

## Future Work

- **Model Training Enhancements**
  - **YOLO11 Training**: Train YOLO11 with the Open Images V7 dataset to improve detection accuracy across more classes.
  - **Specialized Datasets**: Incorporate datasets like Food-101 to enhance detection of specific objects.

- **Performance Optimization**
  - **Parameter Tuning**: Fine-tune confidence thresholds and NMS IoU thresholds.
  - **Processing Speed**: Implement multiprocessing or GPU acceleration for faster processing.

- **Visualization Improvements**
  - **Labeling Algorithms**: Refine label placement and scaling algorithms.

- **Optical Character Recognition (OCR) for Text Detection**
  - **Integration of OCR Models**: Implement OCR to detect and recognize text within images.
  - **Applications**: Enhance the ability to process images containing important textual information, such as signs or documents.

- **Image Captioning with Lightweight Language Models**
  - **Descriptive Capabilities**: Use language models to generate descriptions of the content within images.
  - **User Experience**: Provide users with textual summaries of images, aiding in understanding and accessibility.

- **Detection of Poisoned Images (Backdoor Patterns)**
  - **UnivBD Integration**: Implement Universal Backdoor Detection (UnivBD) techniques to identify and flag poisoned images.
  - **Security Enhancement**: Protect the system from adversarial attacks and ensure data integrity.

- **Advanced Digital Image Processing Techniques**
  - **Contrast Enhancement**: Apply techniques like histogram equalization to improve visibility of hard-to-see components.
  - **Edge Detection and Sharpening**: Enhance image features to facilitate better detection results.
  - **Noise Reduction**: Implement filters to reduce image noise, improving overall detection accuracy.

- **CLIP Integration for Enhanced Object and Contextual Detection**
  - **Contextual Identification**: Integrate OpenAI's CLIP to allow text-based identification of objects by matching image regions with natural language descriptions. This will help refine detection by linking objects to their surrounding context.
  - **Flexible Object Detection**: CLIP's zero-shot capabilities can expand the detection scope to identify additional objects or classes that aren't in the trained model, offering flexibility in diverse detection tasks.
  - **Improved Search and Retrieval**: Utilize CLIP’s capabilities to match image regions with user-defined textual prompts, enabling efficient search and retrieval for specific object types or attributes across processed images.
  - **Cross-Modal Enhancements**: Pair CLIP's text-image matching with existing object detection results to add context-aware labels or captions, enriching visualization and making the interface more informative for users.

  