# processing.py

import logging
from utils.model_loader import ModelLoader
from utils.image_processing import ImageProcessor

import sys
import os
import contextlib

# Define a context manager to suppress stdout and stderr
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def main_processing(config, queue=None):
    """Main function to execute the image processing pipeline."""
    try:
        if queue:
            queue.put(("Loading models...", 0.0))

        # Suppress output for model loading
        with suppress_output():
            large_model = ModelLoader.load_sahi_model(config.YOLO_LARGE_MODEL_PATH, config.CONFIDENCE_THRESHOLD)
            small_model = ModelLoader.load_yolo_model(config.YOLO_SMALL_MODEL_PATH)

        if queue:
            queue.put(("Loaded all models", 0.5))

        # Continue with the rest of processing as usual
        # Define update_callback as a lambda that puts messages into the queue
        if queue:
            update_callback = lambda msg, prog: queue.put((msg, prog))
        else:
            update_callback = None  # Fallback if queue is not provided

        processor = ImageProcessor(config, update_callback=update_callback)
        processor.process_images(large_model, small_model)

        if queue:
            queue.put(("Processing completed.", 1.0))

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        if queue:
            queue.put((f"Error: {e}", 1.0))
        raise
