# config.py

import json
import os
from pathlib import Path
from typing import Tuple, List


class Config:
    """Configuration settings for the image processing pipeline."""
    
    # Parameters for grouping overlapping boxes
    KERNEL_SIZE = 10
    DILATION_ITERATIONS = 2

    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.load_config()

        # Directories (default to current directory)
        self.INPUT_DIRECTORY: Path = Path(self.settings.get("INPUT_DIRECTORY", Path.cwd() / "Images"))
        self.OUTPUT_DIRECTORY: Path = None  # Will be set dynamically based on input directory

        # Set model paths relative to the executable location
        base_path = Path(os.path.dirname(__file__))  # Location of the .exe file
        self.settings["YOLO_LARGE_MODEL_PATH"] = str(base_path / "models/yolo11x.pt")
        self.settings["YOLO_SMALL_MODEL_PATH"] = str(base_path / "models/yolov8x-oiv7.pt")

    def load_config(self):
        """Load configuration settings from a JSON file."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.settings = json.load(f)
        else:
            # Default settings if config file doesn't exist
            self.settings = {
                "YOLO_LARGE_MODEL_PATH": "models/yolo11x.pt",
                "YOLO_SMALL_MODEL_PATH": "models/yolov8x-oiv7.pt",
                "IMAGE_SIZE_THRESHOLD": 1500,
                "COMPLEXITY_THRESHOLD": 0.1,
                "TILE_SIZE": 1024,
                "CONFIDENCE_THRESHOLD": 0.3,
                "NMS_IOU_THRESHOLD": 0.45,
                "SUPPORTED_IMAGE_FORMATS": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
                "SCALES": [0.5, 1.0, 1.5],
                "DILATION_ITERATIONS": 1,
                "KERNEL_SIZE": 5,
                "INPUT_DIRECTORY": str(Path.cwd() / "Images"),
                "USE_SAHI": True  # Default to using SAHI
            }
            self.save_config()

    def save_config(self):
        """Save configuration settings to a JSON file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.settings, f, indent=4)

    # Property methods for settings
    @property
    def YOLO_LARGE_MODEL_PATH(self) -> Path:
        return Path(self.settings["YOLO_LARGE_MODEL_PATH"])

    @property
    def YOLO_SMALL_MODEL_PATH(self) -> Path:
        return Path(self.settings["YOLO_SMALL_MODEL_PATH"])

    @property
    def IMAGE_SIZE_THRESHOLD(self) -> int:
        return self.settings["IMAGE_SIZE_THRESHOLD"]

    @IMAGE_SIZE_THRESHOLD.setter
    def IMAGE_SIZE_THRESHOLD(self, value: int):
        self.settings["IMAGE_SIZE_THRESHOLD"] = value
        self.save_config()

    @property
    def CONFIDENCE_THRESHOLD(self) -> float:
        return self.settings["CONFIDENCE_THRESHOLD"]

    @CONFIDENCE_THRESHOLD.setter
    def CONFIDENCE_THRESHOLD(self, value: float):
        self.settings["CONFIDENCE_THRESHOLD"] = value
        self.save_config()

    @property
    def NMS_IOU_THRESHOLD(self) -> float:
        return self.settings["NMS_IOU_THRESHOLD"]

    @NMS_IOU_THRESHOLD.setter
    def NMS_IOU_THRESHOLD(self, value: float):
        self.settings["NMS_IOU_THRESHOLD"] = value
        self.save_config()

    @property
    def USE_SAHI(self) -> bool:
        return self.settings["USE_SAHI"]

    @USE_SAHI.setter
    def USE_SAHI(self, value: bool):
        self.settings["USE_SAHI"] = value
        self.save_config()

    @property
    def SUPPORTED_IMAGE_FORMATS(self) -> Tuple[str, ...]:
        return tuple(self.settings["SUPPORTED_IMAGE_FORMATS"])

    @property
    def SCALES(self) -> List[float]:
        return self.settings["SCALES"]

    @property
    def DILATION_ITERATIONS(self) -> int:
        return self.settings["DILATION_ITERATIONS"]

    @property
    def KERNEL_SIZE(self) -> int:
        return self.settings["KERNEL_SIZE"]

    @property
    def COMPLEXITY_THRESHOLD(self) -> float:
        return self.settings["COMPLEXITY_THRESHOLD"]

    @property
    def TILE_SIZE(self) -> int:
        return self.settings["TILE_SIZE"]
