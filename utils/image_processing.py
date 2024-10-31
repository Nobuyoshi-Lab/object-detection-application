# utils/image_processing.py

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from ultralytics import YOLO
from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil
from torchvision.ops import nms

from config import Config
import torch

import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.getLogger('sahi').setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Imports for metadata handling
import piexif
from piexif import helper
from PIL import Image, ImageDraw, ImageFont
import json


class ImageProcessor:
    """Class responsible for processing images."""

    def __init__(self, config: Config, update_callback=None, cancel_event=None):
        self.config = config
        self.update_callback = update_callback
        self.cancel_event = cancel_event

    def process_images(self, large_model: Yolov8DetectionModel, small_model: YOLO):
        """Process all images in the input directory."""
        image_files = self.get_image_files(self.config.INPUT_DIRECTORY, self.config.SUPPORTED_IMAGE_FORMATS)
        if not image_files:
            message = "No images found to process."
            logging.warning(message)
            if self.update_callback:
                self.update_callback(f"Error: {message}", 1.0)
            return

        self.create_directory(self.config.OUTPUT_DIRECTORY)

        total_images = len(image_files)
        for idx, image_file in enumerate(image_files):
            if self.cancel_event and self.cancel_event.is_set():
                if self.update_callback:
                    self.update_callback("Processing cancelled by user.", 1.0)
                logging.info("Processing cancelled by user.")
                return

            try:
                # Progress ranges from 0.5 to 1.0 during image processing
                progress = 0.5 + ((idx + 1) / total_images) * 0.5
                if self.update_callback:
                    self.update_callback(f"Processing: {image_file.name}", progress)
                self.process_single_image(image_file, large_model, small_model)
            except Exception as e:
                error_message = f"Error processing image {image_file}: {e}"
                logging.error(error_message)
                if self.update_callback:
                    self.update_callback(f"Error: {error_message}", progress)
        if self.update_callback:
            self.update_callback("Processing completed.", 1.0)
        logging.info("Processing completed.")

    def process_single_image(self, image_file: Path, large_model: Yolov8DetectionModel, small_model: YOLO):
        """Process a single image."""
        if self.cancel_event and self.cancel_event.is_set():
            logging.info("Processing cancelled by user.")
            return

        try:
            # Read image using PIL to avoid color space issues
            img_pil = Image.open(image_file).convert('RGB')
        except FileNotFoundError as fnf_error:
            logging.error(fnf_error)
            if self.update_callback:
                self.update_callback(f"Error: {fnf_error}", 1.0)
            return
        except Exception as e:
            logging.error(f"Unexpected error reading image {image_file}: {e}")
            if self.update_callback:
                self.update_callback(f"Error: {e}", 1.0)
            return

        width, height = img_pil.size

        try:
            if height > self.config.IMAGE_SIZE_THRESHOLD or width > self.config.IMAGE_SIZE_THRESHOLD:
                # Assess complexity
                complexity = self.assess_complexity(img_pil)
                if complexity > self.config.COMPLEXITY_THRESHOLD and self.config.USE_SAHI:
                    # High complexity and SAHI enabled, use SAHI with slicing
                    boxes, classes, labels, img_processed = self.process_large_image(large_model, image_file, img_pil)
                else:
                    # Low complexity or SAHI disabled, process without slicing
                    boxes, classes, labels, img_processed = self.process_small_image(small_model, img_pil, image_file.name)
            else:
                boxes, classes, labels, img_processed = self.process_small_image(small_model, img_pil, image_file.name)
        except Exception as e:
            logging.error(f"Error during detection in image {image_file}: {e}")
            if self.update_callback:
                self.update_callback(f"Error: {e}", 1.0)
            return

        if img_processed is not None:
            try:
                # Save the annotated image using PIL to preserve colors and EXIF
                output_image_path = self.config.OUTPUT_DIRECTORY / image_file.name
                img_processed.save(output_image_path, format='JPEG', quality=95)
                logging.info(f"Saved processed image to {output_image_path}")

                # Add labels to metadata and save with PIL
                self.add_labels_to_metadata(output_image_path, labels, output_image_path.suffix.lower())
                logging.info(f"Embedded metadata into {output_image_path.name}")
            except Exception as e:
                logging.error(f"Failed to save or embed metadata for image {output_image_path}: {e}")
                if self.update_callback:
                    self.update_callback(f"Error: {e}", 1.0)
                return

    def process_large_image(self, model: Yolov8DetectionModel, image_path: Path, original_img: Image.Image):
        """Process a large image using SAHI for slicing and detection."""
        # Get sliced prediction
        prediction = get_sliced_prediction(
            image=original_img,
            detection_model=model,
            slice_height=self.config.TILE_SIZE,
            slice_width=self.config.TILE_SIZE,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type="NMS",
            postprocess_match_threshold=self.config.NMS_IOU_THRESHOLD,
        )

        # Collect detections
        boxes = []
        scores = []
        classes = []
        labels = []
        for obj in prediction.object_prediction_list:
            x1, y1, x2, y2 = obj.bbox.to_xyxy()
            boxes.append([x1, y1, x2, y2])
            scores.append(obj.score.value)
            classes.append(obj.category.id)
            labels.append(obj.category.name)  # Assuming 'name' attribute exists

        if not boxes:
            logging.warning(f"No detections found in image: {image_path}")
            return [], [], [], original_img

        boxes = np.array(boxes)
        scores = np.array(scores)
        classes = np.array(classes)

        # Apply NMS
        boxes, scores, classes = self.apply_nms(boxes, scores, classes, self.config.NMS_IOU_THRESHOLD)

        # Merge detections and mark groups
        boxes, classes, labels = self.merge_detections_by_connected_components(
            original_img.size, boxes, classes, model.model.names
        )

        # Draw bounding boxes and labels on the image using PIL
        img_with_boxes = self.draw_bounding_boxes_pil(original_img.copy(), boxes, classes, labels, model.model.names)

        return boxes, classes, labels, img_with_boxes

    def process_small_image(self, model: YOLO, img_pil: Image.Image, image_name: str):
        """Process a small image using YOLO for detection."""
        # Convert PIL Image to NumPy array for OpenCV processing
        img_np = np.array(img_pil)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Assess complexity
        complexity = self.assess_complexity(img_pil)
        if complexity > self.config.COMPLEXITY_THRESHOLD:
            # Use multi-scale inference without resizing
            detections = []
            labels = []
            for scale in self.config.SCALES:
                # Process the image at original scale
                results = model(img_np, imgsz=self.config.TILE_SIZE, conf=self.config.CONFIDENCE_THRESHOLD)
                for result in results:
                    if not result.boxes:
                        continue
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    detections.append((boxes, scores, classes))
            if not detections:
                logging.warning(f"No detections found in image: {image_name}")
                return [], [], [], img_pil
            # Aggregate detections
            boxes = np.vstack([det[0] for det in detections])
            scores = np.hstack([det[1] for det in detections])
            classes = np.hstack([det[2] for det in detections])
        else:
            # Standard inference without resizing
            results = model(img_np, imgsz=self.config.TILE_SIZE, conf=self.config.CONFIDENCE_THRESHOLD)
            if not results[0].boxes:
                logging.warning(f"No detections found in image: {image_name}")
                return [], [], [], img_pil
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

        if boxes.size == 0:
            logging.warning(f"No detections after NMS in image: {image_name}")
            return [], [], [], img_pil

        # Apply NMS
        boxes, scores, classes = self.apply_nms(boxes, scores, classes, self.config.NMS_IOU_THRESHOLD)

        # Merge detections and mark groups
        boxes, classes, labels = self.merge_detections_by_connected_components(
            img_pil.size, boxes, classes, model.names
        )

        # Draw bounding boxes and labels on the image using PIL
        img_with_boxes = self.draw_bounding_boxes_pil(img_pil.copy(), boxes, classes, labels, model.names)

        return boxes, classes, labels, img_with_boxes

    def add_labels_to_metadata(self, image_path: Path, labels: List[str], image_suffix: str):
        """Add detected labels to the image's EXIF metadata."""
        try:
            if not labels:
                return

            # Only add metadata for JPEG and TIFF images
            if image_suffix.lower() not in ['.jpg', '.jpeg', '.tif', '.tiff']:
                logging.warning(f"Skipping metadata embedding for unsupported image format: {image_path.suffix}")
                return

            # Open the image using PIL
            with Image.open(image_path) as img_pil:
                # Convert to RGB if not already
                if img_pil.mode != 'RGB':
                    img_pil = img_pil.convert('RGB')

                # Load existing EXIF data or initialize a new one
                if 'exif' in img_pil.info:
                    exif_dict = piexif.load(img_pil.info['exif'])
                else:
                    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}

                # Convert labels list to JSON string
                labels_json = json.dumps(labels)

                # Add custom data to the UserComment tag (Tag 37510)
                user_comment = helper.UserComment.dump(labels_json, encoding="unicode")
                exif_dict['Exif'][piexif.ExifIFD.UserComment] = user_comment

                # Convert back to bytes
                exif_bytes = piexif.dump(exif_dict)

                # Save the image with new EXIF data using PIL
                img_pil.save(str(image_path), format=img_pil.format, exif=exif_bytes)
        except AttributeError as ae:
            logging.error(f"AttributeError while adding metadata to {image_path.name}: {ae}")
        except Exception as e:
            logging.error(f"Failed to add metadata to {image_path.name}: {e}")

    def apply_nms(
        self, 
        boxes: np.ndarray, 
        scores: np.ndarray, 
        classes: np.ndarray, 
        iou_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Non-Maximum Suppression per class.
        """
        keep_indices = []
        unique_classes = np.unique(classes)
        for cls in unique_classes:
            cls_indices = np.where(classes == cls)[0]
            cls_boxes = boxes[cls_indices]
            cls_scores = scores[cls_indices]
            if len(cls_boxes) == 0:
                continue

            # Ensure tensors are of type float32
            cls_boxes_tensor = torch.tensor(cls_boxes, dtype=torch.float32)
            cls_scores_tensor = torch.tensor(cls_scores, dtype=torch.float32)

            # Apply NMS
            indices = nms(cls_boxes_tensor, cls_scores_tensor, float(iou_threshold))

            # Convert indices to numpy array and get the corresponding indices
            keep_indices.extend(cls_indices[indices.numpy()])

        if not keep_indices:
            return np.array([]), np.array([]), np.array([])

        final_boxes = boxes[keep_indices]
        final_scores = scores[keep_indices]
        final_classes = classes[keep_indices]
        return final_boxes, final_scores, final_classes

    def merge_detections_by_connected_components(
        self, 
        img_size: Tuple[int, int], 
        boxes: np.ndarray, 
        classes: np.ndarray, 
        model_names: Dict[int, str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Merge detections by connected components analysis and mark groups.
        """
        width, height = img_size
        merged_boxes = []
        merged_classes = []
        labels = []

        unique_classes = np.unique(classes)
        for class_id in unique_classes:
            class_indices = np.where(classes == class_id)[0]
            class_boxes = boxes[class_indices]

            # Create an empty mask
            mask = np.zeros((height, width), dtype=np.uint8)

            # Draw filled rectangles for each detection
            for box in class_boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

            # Dilate the mask to connect nearby regions
            kernel_size = self.config.KERNEL_SIZE  # e.g., 10
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            mask = cv2.dilate(mask, kernel, iterations=self.config.DILATION_ITERATIONS)

            # Find connected components
            num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

            # Initialize a list to keep track of which box belongs to which label
            box_to_label = np.zeros(len(class_boxes), dtype=int)

            # Assign each box to a connected component
            for idx, box in enumerate(class_boxes):
                x1, y1, x2, y2 = box.astype(int)
                # Compute the center of the box
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                # Ensure the center is within image bounds
                x_center = min(max(x_center, 0), width - 1)
                y_center = min(max(y_center, 0), height - 1)
                # Get the connected component label at the center
                label = labels_im[y_center, x_center]
                box_to_label[idx] = label

            # Count boxes per connected component
            unique_labels, counts = np.unique(box_to_label, return_counts=True)

            # Iterate over each connected component
            for label_idx, count in zip(unique_labels, counts):
                if label_idx == 0:
                    continue  # Skip background
                # Find all boxes that belong to this label
                member_indices = np.where(box_to_label == label_idx)[0]
                member_boxes = class_boxes[member_indices]

                # Compute the merged bounding box
                x1 = member_boxes[:, 0].min()
                y1 = member_boxes[:, 1].min()
                x2 = member_boxes[:, 2].max()
                y2 = member_boxes[:, 3].max()
                merged_boxes.append([x1, y1, x2, y2])
                merged_classes.append(class_id)

                # Get the label name and append "(group)" if multiple boxes are merged
                label_name = model_names.get(int(class_id), "Unknown")
                if count > 1:
                    label_name += " (group)"
                labels.append(label_name)

                # Logging the grouping action
                logging.info(f"Merged {count} boxes for class '{label_name}' into [{x1}, {y1}, {x2}, {y2}]")

        if not merged_boxes:
            return np.array([]), np.array([]), []

        return np.array(merged_boxes), np.array(merged_classes), labels

    def draw_bounding_boxes_pil(
        self, 
        img_pil: Image.Image, 
        boxes: np.ndarray, 
        classes: np.ndarray, 
        labels: List[str], 
        model_names: Dict[int, str]
    ) -> Image.Image:
        """
        Draw bounding boxes and labels on a PIL Image.
        """
        draw = ImageDraw.Draw(img_pil)
        try:
            # Use a truetype font with larger size
            font = ImageFont.truetype("arial.ttf", size=20)
        except IOError:
            # Fallback to default font if arial is not available
            font = ImageFont.load_default()

        for box, cls, label in zip(boxes, classes, labels):
            x1, y1, x2, y2 = box.astype(int)
            # Assign a unique color based on class ID
            color = self.get_color_for_class(int(cls))
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Draw label background
            # Calculate text size using textbbox
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            padding = 4
            text_background = [x1, y1, x1 + text_width + padding, y1 + text_height + padding]
            draw.rectangle(text_background, fill=color)
            # Draw label text inside the bounding box
            draw.text((x1 + 2, y1 + 2), label, fill="white", font=font)

        return img_pil

    def get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Assign a unique color to each class."""
        np.random.seed(int(class_id))
        color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
        return color

    def create_directory(self, directory: Path):
        """Create the output directory if it doesn't exist."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"Ensured existence of directory: {directory}")
        except Exception as e:
            logging.error(f"Failed to create directory {directory}: {e}")
            if self.update_callback:
                self.update_callback(f"Error: Failed to create directory {directory}: {e}", 1.0)

    def get_image_files(self, directory: Path, supported_formats: Tuple[str, ...]) -> List[Path]:
        """Retrieve a list of image files from the input directory."""
        if not directory.exists():
            logging.error(f"Input directory {directory} does not exist.")
            return []

        image_files = [file for file in directory.iterdir() if file.suffix.lower() in supported_formats]
        logging.info(f"Found {len(image_files)} images to process.")
        return image_files

    def assess_complexity(self, image: Image.Image) -> float:
        """Assess the complexity of an image based on edge density."""
        gray = image.convert('L')  # Convert to grayscale
        gray_np = np.array(gray)
        edges = cv2.Canny(gray_np, threshold1=50, threshold2=150)
        edge_density = np.sum(edges) / (gray_np.shape[0] * gray_np.shape[1])
        logging.debug(f"Edge density: {edge_density}")
        return edge_density
