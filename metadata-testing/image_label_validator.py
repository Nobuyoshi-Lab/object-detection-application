import os
import json
import logging
import re
from tkinter import filedialog, messagebox
from PIL import Image
import piexif
from piexif import helper
import customtkinter as ctk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Initialize Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Initialize CustomTkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Constants
CONFIG_FILE = "config.json"
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# Synonym dictionary to handle variations
SYNONYMS = {
    "dairy_product": ["muffin", "pastry", "dairy product"],
    "food": [
        "spoon", "cup", "plate", "bowl", "platter", "bottle", "sauce",
        "salad", "mashed potatoes", "grilled chicken", "bottled water",
        "white bread", "squeeze bottle"
    ],
    "person": ["man", "woman", "avatar", "people"],
    "umbrella": ["umbrella"],
    "book": ["book", "books"],
    "chair": ["chair", "chairs"],
    "helmet": ["helmet"],
    "jet_ski": ["jet ski"],
    "plant": ["plant", "plants"],
    "vegetable": ["vegetables", "vegetable"],
    "bamboo": ["bamboo"],
    "surfboard": ["surfboard"],
    "boat": ["boat"],
    "saucer": ["saucer"],
    "coffee_cup": ["coffee cup"],
    "baked_goods": ["baked goods"],
    "kitchen_dining_room_table": ["kitchen & dining room table"],
    "flower": ["flower"],
    "bookcase": ["bookcase"],
    "building": ["building"],
    "shelf": ["shelf"],
    "bagel": ["bagel"],
    "donut": ["donut"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "banana": ["banana"],
    "knife": ["knife"],
    "dining_table": ["dining table"],
    "insect": ["insect"],
    "footwear": ["footwear"],
    "billboard": ["billboard"],
    "computer_interface": ["computer interface"],
    "dal": ["dal"],
    "vegetable_curry": ["vegetable curry"],
    "dipping_sauce": ["dipping sauce"],
    "fried_foods": ["fried foods"],
}

def load_config():
    """Load configuration from CONFIG_FILE."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as file:
                config = json.load(file)
                logging.info("Configuration loaded successfully.")
                return config
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding config file: {e}")
    return {}

def save_config(config):
    """Save configuration to CONFIG_FILE."""
    try:
        with open(CONFIG_FILE, 'w') as file:
            json.dump(config, file, indent=4)
            logging.info("Configuration saved successfully.")
    except Exception as e:
        logging.error(f"Error saving config file: {e}")

def normalize_label(label):
    """Normalize a label for case and synonym matching."""
    label = label.strip().lower()

    # Remove any parenthetical suffixes like ' (group)', ' (groups)', etc.
    label = re.sub(r'\s*\(.*?\)', '', label).strip()

    for key, synonyms in SYNONYMS.items():
        if label in synonyms or label == key:
            return key
    return label

def select_folder(title):
    """Open a dialog to select a directory."""
    return filedialog.askdirectory(title=title)

def select_file(title, filetypes):
    """Open a dialog to select a file."""
    return filedialog.askopenfilename(title=title, filetypes=filetypes)

class LabelComparison:
    """Handles label extraction and metric calculations."""

    def __init__(self, image_folder, json_file):
        self.image_folder = image_folder
        self.json_file = json_file
        self.extracted_labels = {}
        self.user_labels = {}

    def display_metadata(self):
        """Generate a string containing metadata labels for all images."""
        metadata_output = []
        for image_file in os.listdir(self.image_folder):
            if os.path.splitext(image_file)[1].lower() in SUPPORTED_FORMATS:
                image_path = os.path.join(self.image_folder, image_file)
                labels = self.get_labels_from_metadata(image_path)
                if labels:
                    # Keep original labels for display
                    metadata_output.append(f"{image_file} - Metadata: {labels}")
        return "\n".join(metadata_output) if metadata_output else "No metadata found in the selected images."

    def load_user_defined_labels(self):
        """Load user-defined labels from a JSON file."""
        try:
            with open(self.json_file, 'r') as file:
                data = json.load(file)
                self.user_labels = {
                    img: [normalize_label(label) for label in labels]
                    for img, labels in data.items()
                }
            logging.info(f"Loaded user labels: {self.user_labels}")
        except FileNotFoundError:
            logging.error("User-defined JSON file not found.")
            messagebox.showerror("File Not Found", "The selected JSON file does not exist.")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON file: {e}")
            messagebox.showerror("Invalid JSON", f"Error decoding JSON file: {e}")
        except Exception as e:
            logging.error(f"Unexpected error loading user-defined labels: {e}")
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def extract_labels_from_folder(self):
        """Extract labels from image metadata in the selected folder."""
        for image_file in os.listdir(self.image_folder):
            if os.path.splitext(image_file)[1].lower() in SUPPORTED_FORMATS:
                image_path = os.path.join(self.image_folder, image_file)
                labels = self.get_labels_from_metadata(image_path)
                if labels:
                    normalized_labels = set(normalize_label(label) for label in labels)
                    self.extracted_labels[image_file] = normalized_labels
        logging.info(f"Extracted labels: {self.extracted_labels}")

    @staticmethod
    def get_labels_from_metadata(image_path):
        """Extract labels from the EXIF UserComment field of an image."""
        try:
            with Image.open(image_path) as img:
                exif_data = img.info.get('exif')
                if not exif_data:
                    logging.warning(f"No EXIF data found for {image_path}.")
                    return []

                exif_dict = piexif.load(exif_data)
                user_comment = exif_dict['Exif'].get(piexif.ExifIFD.UserComment)
                if not user_comment:
                    logging.warning(f"No UserComment found in EXIF data for {image_path}.")
                    return []

                # Decode UserComment based on encoding
                decoded_comment = helper.UserComment.load(user_comment)
                labels = json.loads(decoded_comment)
                if not isinstance(labels, list):
                    logging.warning(f"UserComment in {image_path} is not a list.")
                    return []
                return labels
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding error for {image_path}: {e}")
        except Exception as e:
            logging.error(f"Error reading metadata from {image_path}: {e}")
        return []

    def calculate_metrics(self):
        """Calculate multi-label classification metrics."""
        # Collect all unique labels from user and extracted labels
        all_labels = set()
        for labels in self.user_labels.values():
            all_labels.update(labels)
        for labels in self.extracted_labels.values():
            all_labels.update(labels)
        all_labels = sorted(all_labels)

        if not all_labels:
            logging.warning("No labels found for metric calculation.")
            return {}, {}

        mlb = MultiLabelBinarizer()
        # Fit the binarizer on all possible labels to ensure consistency
        mlb.fit([all_labels])

        # Prepare true and predicted label lists
        y_true = []
        y_pred = []

        all_images = set(self.user_labels.keys()).union(self.extracted_labels.keys())
        for img in all_images:
            true_labels = set(self.user_labels.get(img, []))
            pred_labels = set(self.extracted_labels.get(img, []))

            y_true.append(list(true_labels))
            y_pred.append(list(pred_labels))

        y_true_bin = mlb.transform(y_true)
        y_pred_bin = mlb.transform(y_pred)

        # Calculate Strict Accuracy (Subset Accuracy)
        strict_accuracy = accuracy_score(y_true_bin, y_pred_bin) * 100

        # Calculate Presence-Based Metrics using Micro Averaging
        precision_micro = precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0) * 100
        recall_micro = recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0) * 100
        f1_micro = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0) * 100

        # Calculate Presence-Based Metrics using Macro Averaging
        precision_macro = precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0) * 100
        recall_macro = recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0) * 100
        f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0) * 100

        metrics_strict = {
            'accuracy': strict_accuracy,
        }

        # Use formatted keys to match GUI expectations
        metrics_presence = {
            'Precision (Micro)': precision_micro,
            'Recall (Micro)': recall_micro,
            'F1-score (Micro)': f1_micro,
            'Precision (Macro)': precision_macro,
            'Recall (Macro)': recall_macro,
            'F1-score (Macro)': f1_macro,
        }

        logging.info(f"Strict Metrics: {metrics_strict}")
        logging.info(f"Presence-Based Metrics: {metrics_presence}")

        return metrics_strict, metrics_presence

class ObjectDetectionApp(ctk.CTk):
    """Main application class for Object Detection Label Comparison."""

    def __init__(self):
        super().__init__()
        self.title("Object Detection Label Comparison")
        self.geometry("900x700")
        self.resizable(False, False)

        self.config_data = load_config()
        self.image_folder = ctk.StringVar(value=self.config_data.get("image_folder", ""))
        self.json_file = ctk.StringVar(value=self.config_data.get("json_file", ""))

        self.create_widgets()

    def create_widgets(self):
        """Create and layout all GUI components."""
        # Frame for Image Folder Selection
        frame_folder = self.create_selection_frame(
            parent=self,
            label_text="Select Image Folder:",
            variable=self.image_folder,
            browse_command=self.select_image_folder
        )
        frame_folder.pack(pady=10, padx=20, fill="x")

        # Frame for JSON File Selection
        frame_json = self.create_selection_frame(
            parent=self,
            label_text="Select JSON File:",
            variable=self.json_file,
            browse_command=self.select_json_file,
            filetypes=[("JSON files", "*.json")]
        )
        frame_json.pack(pady=10, padx=20, fill="x")

        # Frame for Action Buttons
        frame_actions = ctk.CTkFrame(self)
        frame_actions.pack(pady=10, padx=20, fill="x")

        button_run = ctk.CTkButton(
            frame_actions,
            text="Run Comparison",
            command=self.run_comparison,
            width=200
        )
        button_run.pack(side="left", padx=10)

        button_display = ctk.CTkButton(
            frame_actions,
            text="Display Metadata",
            command=self.display_metadata,
            width=200
        )
        button_display.pack(side="right", padx=10)

        # Frame for Metrics Display
        frame_metrics = ctk.CTkFrame(self)
        frame_metrics.pack(pady=10, padx=20, fill="x")

        self.create_metrics_display(frame_metrics)

        # Frame for Metadata Display with Scrollbar
        frame_metadata = ctk.CTkFrame(self)
        frame_metadata.pack(pady=10, padx=20, fill="both", expand=True)

        # Enhanced "Metadata Output:" Label as a Section Title
        label_metadata_title = ctk.CTkLabel(
            frame_metadata,
            text="Metadata Output:",
            font=("Arial", 16, "bold")
        )
        label_metadata_title.grid(row=0, column=0, sticky="w", padx=10, pady=(0,5))

        # CTkTextbox for Metadata Display
        self.metadata_output_textbox = ctk.CTkTextbox(
            frame_metadata,
            wrap="word",
            state="disabled"  # Start as disabled
        )
        self.metadata_output_textbox.grid(row=1, column=0, sticky="nsew", padx=(10,0), pady=5)

        # CTkScrollbar associated with the CTkTextbox
        scrollbar = ctk.CTkScrollbar(
            frame_metadata,
            orientation="vertical",
            command=self.metadata_output_textbox.yview
        )
        scrollbar.grid(row=1, column=1, sticky="ns", padx=(0,10), pady=5)

        # Configure the CTkTextbox to use the scrollbar
        self.metadata_output_textbox.configure(yscrollcommand=scrollbar.set)

        # Configure grid weights for proper resizing
        frame_metadata.grid_rowconfigure(1, weight=1)
        frame_metadata.grid_columnconfigure(0, weight=1)

    def create_selection_frame(self, parent, label_text, variable, browse_command, filetypes=None):
        """
        Helper method to create a selection frame for folder or file selection.
        :param parent: Parent widget.
        :param label_text: Text for the label.
        :param variable: StringVar associated with the entry.
        :param browse_command: Command to execute on browse button click.
        :param filetypes: File types for file selection dialog.
        :return: Frame containing the selection widgets.
        """
        frame = ctk.CTkFrame(parent)
        frame.grid_columnconfigure(1, weight=1)  # Make the entry expand

        label = ctk.CTkLabel(frame, text=label_text, anchor="w")
        label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        entry = ctk.CTkEntry(frame, textvariable=variable, state="readonly")
        entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        button_browse = ctk.CTkButton(
            frame,
            text="Browse",
            command=lambda: browse_command(filetypes) if filetypes else browse_command(),
            width=100
        )
        button_browse.grid(row=0, column=2, padx=5, pady=5)

        return frame

    def create_metrics_display(self, parent):
        """
        Create the metrics display section with headers and metric labels.
        :param parent: Parent widget.
        """
        # Headers
        header_strict = ctk.CTkLabel(parent, text="Strict Matching Metrics", font=("Arial", 14, "bold"))
        header_strict.grid(row=0, column=0, padx=20, pady=5, sticky="w")

        header_presence = ctk.CTkLabel(parent, text="Presence-Based Matching Metrics", font=("Arial", 14, "bold"))
        header_presence.grid(row=0, column=1, padx=20, pady=5, sticky="w")

        # Metric Labels
        self.metrics_labels_strict = {}
        self.metrics_labels_presence = {}
        # Only Accuracy is meaningful for strict metrics
        label_strict_accuracy = ctk.CTkLabel(parent, text="Accuracy:", font=("Arial", 12))
        label_strict_accuracy.grid(row=1, column=0, sticky="w", padx=20, pady=2)

        label_value_strict_accuracy = ctk.CTkLabel(parent, text="N/A", font=("Arial", 12))
        label_value_strict_accuracy.grid(row=1, column=0, sticky="e", padx=20, pady=2)
        self.metrics_labels_strict["accuracy"] = label_value_strict_accuracy

        # Presence-Based Metrics
        presence_metrics = [
            "Precision (Micro)", "Recall (Micro)", "F1-score (Micro)",
            "Precision (Macro)", "Recall (Macro)", "F1-score (Macro)"
        ]

        for i, metric in enumerate(presence_metrics, start=2):
            label_metric = ctk.CTkLabel(parent, text=f"{metric}:", font=("Arial", 12))
            label_metric.grid(row=i, column=1, sticky="w", padx=20, pady=2)

            label_value = ctk.CTkLabel(parent, text="N/A", font=("Arial", 12))
            label_value.grid(row=i, column=1, sticky="e", padx=20, pady=2)
            self.metrics_labels_presence[metric] = label_value

    def display_metadata(self):
        """Display metadata of images in the selected folder."""
        image_folder = self.image_folder.get()

        if not os.path.isdir(image_folder):
            messagebox.showerror("Invalid Folder", "Please select a valid image folder.")
            logging.error("Invalid folder selection.")
            return

        comparison = LabelComparison(image_folder, "")
        metadata_output = comparison.display_metadata()

        # Update the textbox
        self.metadata_output_textbox.configure(state="normal")
        self.metadata_output_textbox.delete("1.0", "end")
        self.metadata_output_textbox.insert("1.0", metadata_output)
        self.metadata_output_textbox.configure(state="disabled")

        # Inform the user
        if metadata_output:
            messagebox.showinfo("Metadata Displayed", "Metadata has been displayed successfully.")
        else:
            messagebox.showinfo("No Metadata", "No metadata found in the selected images.")

    def select_image_folder(self, filetypes=None):
        """Handle the image folder selection."""
        folder = select_folder("Select Folder with Images")
        if folder:
            self.image_folder.set(folder)
            self.config_data["image_folder"] = folder
            save_config(self.config_data)
            logging.info(f"Image folder selected: {folder}")

    def select_json_file(self, filetypes=None):
        """Handle the JSON file selection."""
        file_path = select_file("Select JSON File with User-defined Labels", [("JSON files", "*.json")]) if not filetypes else select_file("Select File", filetypes)
        if file_path:
            self.json_file.set(file_path)
            self.config_data["json_file"] = file_path
            save_config(self.config_data)
            logging.info(f"JSON file selected: {file_path}")

    def run_comparison(self):
        """Run the label comparison and update metrics."""
        image_folder = self.image_folder.get()
        json_file = self.json_file.get()

        if not os.path.isdir(image_folder):
            messagebox.showerror("Invalid Folder", "Please select a valid image folder.")
            self.reset_metrics()
            return

        if not os.path.isfile(json_file):
            messagebox.showerror("Invalid File", "Please select a valid JSON file.")
            self.reset_metrics()
            return

        # Disable buttons to prevent multiple clicks
        self.disable_buttons()

        try:
            comparison = LabelComparison(image_folder, json_file)
            comparison.load_user_defined_labels()
            comparison.extract_labels_from_folder()

            metrics_strict, metrics_presence = comparison.calculate_metrics()

            if not metrics_strict and not metrics_presence:
                messagebox.showwarning("No Metrics", "No metrics could be calculated. Please check your data.")
                self.reset_metrics()
                return

            # Update the GUI with Strict Metrics
            strict_accuracy = metrics_strict.get('accuracy', None)
            if strict_accuracy is not None:
                self.metrics_labels_strict["accuracy"].configure(text=f"{strict_accuracy:.2f}%")
            else:
                self.metrics_labels_strict["accuracy"].configure(text="N/A")
                logging.warning("Strict accuracy not calculated.")

            # Update the GUI with Presence-Based Metrics
            for metric, value in metrics_presence.items():
                if metric in self.metrics_labels_presence:
                    self.metrics_labels_presence[metric].configure(text=f"{value:.2f}%")
                else:
                    logging.warning(f"Metric key '{metric}' not found in metrics_labels_presence.")

            messagebox.showinfo("Comparison Completed", "Label comparison metrics have been updated.")

            # Optional: Generate and display plots
            self.generate_plots(metrics_strict, metrics_presence)

        except Exception as e:
            logging.error(f"Error during comparison: {e}")
            messagebox.showerror("Error", f"An error occurred during comparison: {e}")
        finally:
            # Re-enable buttons
            self.enable_buttons()

    def reset_metrics(self):
        """Reset all metric labels to default."""
        self.metrics_labels_strict["accuracy"].configure(text="N/A")
        for label in self.metrics_labels_presence.values():
            label.configure(text="N/A")

    def disable_buttons(self):
        """Disable action buttons during processing."""
        for child in self.winfo_children():
            if isinstance(child, ctk.CTkFrame):
                for subchild in child.winfo_children():
                    if isinstance(subchild, ctk.CTkButton):
                        subchild.configure(state="disabled")

    def enable_buttons(self):
        """Enable action buttons after processing."""
        for child in self.winfo_children():
            if isinstance(child, ctk.CTkFrame):
                for subchild in child.winfo_children():
                    if isinstance(subchild, ctk.CTkButton):
                        subchild.configure(state="normal")

    def generate_plots(self, metrics_strict, metrics_presence):
        """Generate and display plots for the metrics."""
        # Prepare data for bar chart
        metrics = ['Strict Accuracy'] + list(metrics_presence.keys())
        values = [metrics_strict.get('accuracy', 0)] + list(metrics_presence.values())

        plt.figure(figsize=(10,6))
        sns.barplot(x=metrics, y=values, palette='viridis')
        plt.xlabel('Metrics')
        plt.ylabel('Scores (%)')
        plt.title('Object Detection Label Comparison Metrics')
        plt.ylim(0, 100)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # Optional: Save the plot
        # plt.savefig('metrics_plot.png')

if __name__ == "__main__":
    app = ObjectDetectionApp()
    app.mainloop()
