# utils/model_loader.py

from pathlib import Path
from ultralytics import YOLO
from sahi.models.yolov8 import Yolov8DetectionModel
import torch


class ModelLoader:
    """Class responsible for loading models."""

    @staticmethod
    def load_sahi_model(model_path: Path, confidence_threshold: float) -> Yolov8DetectionModel:
        """Load the SAHI detection model."""
        detection_model = Yolov8DetectionModel(
            model_path=str(model_path),
            confidence_threshold=confidence_threshold,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return detection_model

    @staticmethod
    def load_yolo_model(model_path: Path) -> YOLO:
        """Load the YOLO model."""
        model = YOLO(str(model_path))
        return model
