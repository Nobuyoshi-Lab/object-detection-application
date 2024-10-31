# gui.py

import multiprocessing
from pathlib import Path
import os
import platform
import customtkinter as ctk
from tkinter import messagebox

from processing import main_processing

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

LIGHT_BLUE = "#B0E0E6"
LIGHT_GREEN = "#90EE90"

import emoji

class App:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.is_processing = False
        self.process = None  # Reference to the processing process
        self.queue = multiprocessing.Queue()  # Queue for inter-process communication

        # Set appearance mode based on system's default before creating widgets
        self.set_initial_theme()

        self.create_widgets()
        self.check_queue()

    def set_initial_theme(self):
        """
        Set the initial theme based on the system's appearance.
        If you want to restrict to only Light and Dark without System option,
        uncomment the detection logic below.
        """
        # Option 1: Use System Appearance
        ctk.set_appearance_mode("System")
        
        # Option 2: Manually detect and set Light or Dark
        # Uncomment the following lines if you prefer manual detection
        """
        if platform.system() == "Windows":
            # Windows 10 and above can have theme info in registry or using ctypes
            import ctypes
            try:
                # Windows 10 build 17763 and above
                preference = ctypes.windll.dwmapi.DwmGetWindowAttribute
                # This is a placeholder; actual implementation requires specific API calls
                # For simplicity, default to Light
                system_mode = "Light"
            except:
                system_mode = "Light"
        elif platform.system() == "Darwin":
            # macOS detection
            from subprocess import check_output
            try:
                result = check_output(['defaults', 'read', '-g', 'AppleInterfaceStyle']).decode().strip()
                system_mode = "Dark" if result == "Dark" else "Light"
            except:
                system_mode = "Light"
        else:
            # For Linux or others, default to Light
            system_mode = "Light"

        ctk.set_appearance_mode(system_mode)
        """

    def create_widgets(self):
        self.root.title("Object Detection Application")
        self.root.geometry("500x550")
        self.root.resizable(False, False)

        # Set default font
        default_font = ("Cascadia Mono", 12, "bold")

        # Input Directory
        self.input_dir_label = ctk.CTkLabel(self.root, text="Input Images Directory:", font=default_font)
        self.input_dir_label.grid(row=0, column=0, sticky='w', padx=10, pady=10)
        self.input_dir_entry = ctk.CTkEntry(self.root, width=300, font=default_font)
        self.input_dir_entry.insert(0, str(self.config.INPUT_DIRECTORY))
        self.input_dir_entry.grid(row=0, column=1, padx=10, pady=10, columnspan=2, sticky='we')
        self.browse_button = ctk.CTkButton(self.root, text="Browse", command=self.browse_input_dir, width=100, font=default_font)
        self.browse_button.grid(row=1, column=1, padx=10, pady=10, sticky='w')
        self.open_output_button = ctk.CTkButton(self.root, text="Open Output Folder", command=self.open_output_folder, width=150, font=default_font)
        self.open_output_button.grid(row=1, column=2, padx=10, pady=10, sticky='w')

        # Confidence Threshold
        self.conf_label = ctk.CTkLabel(self.root, text=f"Confidence Threshold: {self.config.CONFIDENCE_THRESHOLD:.2f}", font=default_font)
        self.conf_label.grid(row=2, column=0, padx=10, pady=10, sticky='w')
        self.confidence_scale = ctk.CTkSlider(self.root, from_=0.0, to=1.0, number_of_steps=100,
                                              command=self.update_conf_label)
        self.confidence_scale.set(self.config.CONFIDENCE_THRESHOLD)
        self.confidence_scale.grid(row=2, column=1, columnspan=2, padx=10, pady=10, sticky='we')

        # NMS IOU Threshold
        self.nms_label = ctk.CTkLabel(self.root, text=f"NMS IOU Threshold: {self.config.NMS_IOU_THRESHOLD:.2f}", font=default_font)
        self.nms_label.grid(row=3, column=0, padx=10, pady=10, sticky='w')
        self.nms_scale = ctk.CTkSlider(self.root, from_=0.0, to=1.0, number_of_steps=100,
                                       command=self.update_nms_label)
        self.nms_scale.set(self.config.NMS_IOU_THRESHOLD)
        self.nms_scale.grid(row=3, column=1, columnspan=2, padx=10, pady=10, sticky='we')

        # Image Size Threshold
        self.image_size_label = ctk.CTkLabel(self.root, text="Image Size Threshold:", font=default_font)
        self.image_size_label.grid(row=4, column=0, padx=10, pady=10, sticky='w')
        self.image_size_entry = ctk.CTkEntry(self.root, font=default_font)
        self.image_size_entry.insert(0, str(self.config.IMAGE_SIZE_THRESHOLD))
        self.image_size_entry.grid(row=4, column=1, columnspan=2, padx=10, pady=10, sticky='we')

        # SAHI Option
        self.sahi_var = ctk.BooleanVar(value=self.config.USE_SAHI)
        self.sahi_checkbox = ctk.CTkCheckBox(self.root, text="Use SAHI for Large Images", variable=self.sahi_var, font=default_font)
        self.sahi_checkbox.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky='w')

        # Center Frame for Status, Progress Bar, and Start/Cancel Buttons
        self.center_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.center_frame.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

        # Status Label
        status_bg_color, text_color = self._get_status_colors()
        self.status_label = ctk.CTkLabel(
            self.center_frame,
            text="Status: Idle",
            text_color=text_color,
            fg_color=status_bg_color,
            font=default_font,
            corner_radius=5,
            padx=10,
            pady=5
        )
        self.status_label.pack(pady=5)

        # Progress Bar
        self.progress_var = ctk.DoubleVar()
        self.progress_bar = ctk.CTkProgressBar(self.center_frame, variable=self.progress_var, width=400)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=5)

        # Start/Cancel Button
        self.start_cancel_button = ctk.CTkButton(
            self.center_frame,
            text="Start Processing",
            command=self.start_or_cancel_processing,
            font=default_font
        )
        self.start_cancel_button.pack(pady=10)
        
        # Store default button configuration
        self.default_button_text = "Start Processing"
        self.default_button_fg_color = self.start_cancel_button.cget("fg_color")
        
        # Expand center frame to fill available space
        self.root.grid_rowconfigure(6, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        # Adjusted theme_frame and buttons
        self.theme_frame = ctk.CTkFrame(self.root, fg_color="transparent", width=150, height=60)
        self.theme_frame.grid(row=7, column=0, padx=10, pady=10, sticky='sw')

        # Configure font and button settings
        emoji_font = ("Segoe UI Emoji", 14)
        button_config = {
            "width": 40,
            "height": 40,
            "corner_radius": 20,
            "font": emoji_font
        }

        # Add theme buttons
        self.default_theme_button = ctk.CTkButton(
            self.theme_frame,
            text=emoji.emojize(":laptop:"),
            command=lambda: self.change_theme("System"),
            **button_config
        )
        self.default_theme_button.grid(row=0, column=0, padx=5, pady=5)

        self.light_theme_button = ctk.CTkButton(
            self.theme_frame,
            text=emoji.emojize(":sun:"),
            command=lambda: self.change_theme("Light"),
            **button_config
        )
        self.light_theme_button.grid(row=0, column=1, padx=5, pady=5)

        self.dark_theme_button = ctk.CTkButton(
            self.theme_frame,
            text=emoji.emojize(":crescent_moon:"),
            command=lambda: self.change_theme("Dark"),
            **button_config
        )
        self.dark_theme_button.grid(row=0, column=2, padx=5, pady=5)

        # Allow flexible resizing
        self.theme_frame.grid_rowconfigure(0, weight=1)
        self.theme_frame.grid_columnconfigure(0, weight=1)
        self.theme_frame.grid_columnconfigure(1, weight=1)
        self.theme_frame.grid_columnconfigure(2, weight=1)

    def _get_status_colors(self):
        bg_color = "#444444"
        text_color = "white"
        return bg_color, text_color

    def browse_input_dir(self):
        dir_selected = ctk.filedialog.askdirectory(initialdir=str(self.config.INPUT_DIRECTORY))
        if dir_selected:
            self.input_dir_entry.delete(0, ctk.END)
            self.input_dir_entry.insert(0, dir_selected)
            self.config.INPUT_DIRECTORY = Path(dir_selected)
            self.config.settings["INPUT_DIRECTORY"] = dir_selected
            self.config.save_config()

    def open_output_folder(self):
        if self.config.OUTPUT_DIRECTORY and self.config.OUTPUT_DIRECTORY.exists():
            path = str(self.config.OUTPUT_DIRECTORY)
            try:
                if platform.system() == "Windows":
                    os.startfile(path)
                elif platform.system() == "Darwin":  # macOS
                    os.system(f'open "{path}"')
                else:  # Linux and others
                    os.system(f'xdg-open "{path}"')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open output folder: {e}")
        else:
            messagebox.showinfo("Info", "Processed images folder does not exist yet.")

    def change_theme(self, new_theme):
        # If you have removed the System theme, handle accordingly
        if new_theme not in ["Light", "Dark", "System"]:
            new_theme = "Light"  # Fallback to Light

        ctk.set_appearance_mode(new_theme)
        # Update status label colors based on new theme
        bg_color, text_color = self._get_status_colors()
        self.status_label.configure(fg_color=bg_color, text_color=text_color)

    def update_conf_label(self, value):
        self.conf_label.configure(text=f"Confidence Threshold: {float(value):.2f}")
        self.config.CONFIDENCE_THRESHOLD = float(value)

    def update_nms_label(self, value):
        self.nms_label.configure(text=f"NMS IOU Threshold: {float(value):.2f}")
        self.config.NMS_IOU_THRESHOLD = float(value)

    def start_or_cancel_processing(self):
        if not self.is_processing:
            self.start_processing()
        else:
            self.cancel_processing()

    def start_processing(self):
        if self.is_processing:
            messagebox.showwarning("Processing", "Processing is already running.")
            return

        # Update config with user inputs
        self.config.INPUT_DIRECTORY = Path(self.input_dir_entry.get())
        self.config.settings["INPUT_DIRECTORY"] = str(self.config.INPUT_DIRECTORY)
        self.config.save_config()

        try:
            self.config.CONFIDENCE_THRESHOLD = float(self.confidence_scale.get())
            self.config.NMS_IOU_THRESHOLD = float(self.nms_scale.get())
            self.config.IMAGE_SIZE_THRESHOLD = int(self.image_size_entry.get())
            self.config.USE_SAHI = self.sahi_var.get()
        except ValueError as ve:
            messagebox.showerror("Invalid Input", f"Please enter valid values: {ve}")
            return

        # Automatically set OUTPUT_DIRECTORY to a new folder inside INPUT_DIRECTORY
        self.config.OUTPUT_DIRECTORY = self.config.INPUT_DIRECTORY / "Processed_Images"

        # Change Start button to Cancel
        self.start_cancel_button.configure(text="Cancel", fg_color="red")

        # Disable controls
        self.disable_controls()

        # Update status
        self.is_processing = True
        self.status_label.configure(text="Status: Processing...", text_color=LIGHT_BLUE)
        self.progress_var.set(0)
        self.progress_bar.update()

        # Start processing in a separate process
        self.process = multiprocessing.Process(
            target=main_processing,
            args=(self.config, self.queue)
        )
        self.process.start()

    def reset_ui(self):
        # Reset Start/Cancel Button to default
        self.start_cancel_button.configure(
            text=self.default_button_text,
            fg_color=self.default_button_fg_color
        )
        
        # Reset Status Label
        bg_color, text_color = self._get_status_colors()
        self.status_label.configure(
            text="Status: Idle",
            fg_color=bg_color,
            text_color=text_color
        )
        
        # Reset Progress Bar
        self.progress_var.set(0)
        self.progress_bar.update()
        
        # Reset processing state
        self.is_processing = False
        self.process = None
        
        # Enable all controls
        self.enable_controls()
    
    def cancel_processing(self):
        if self.process and self.process.is_alive():
            self.process.terminate()  # Forcefully terminate the process
            self.process.join()
            self.process = None
            self.reset_ui()
            self.status_label.configure(text="Status: Cancelled", text_color="orange")
            messagebox.showinfo("Cancelled", "Image processing was cancelled.")
    
    def disable_controls(self):
        self.input_dir_entry.configure(state='disabled')
        self.browse_button.configure(state='disabled')
        self.open_output_button.configure(state='disabled')
        self.confidence_scale.configure(state='disabled')
        self.nms_scale.configure(state='disabled')
        self.image_size_entry.configure(state='disabled')
        self.sahi_checkbox.configure(state='disabled')
        self.default_theme_button.configure(state='disabled')
        self.light_theme_button.configure(state='disabled')
        self.dark_theme_button.configure(state='disabled')

    def enable_controls(self):
        self.input_dir_entry.configure(state='normal')
        self.browse_button.configure(state='normal')
        self.open_output_button.configure(state='normal')
        self.confidence_scale.configure(state='normal')
        self.nms_scale.configure(state='normal')
        self.image_size_entry.configure(state='normal')
        self.sahi_checkbox.configure(state='normal')
        self.default_theme_button.configure(state='normal')
        self.light_theme_button.configure(state='normal')
        self.dark_theme_button.configure(state='normal')

    def check_queue(self):
        """Check the multiprocessing queue for status updates."""
        while not self.queue.empty():
            try:
                message, progress = self.queue.get_nowait()
                self.update_status(message, progress)
                if message == "Processing completed.":
                    self.processing_done()
                elif message.startswith("Error:"):
                    self.processing_error(message)
            except Exception as e:
                logging.error(f"Error reading from queue: {e}")
        self.root.after(100, self.check_queue)

    def processing_done(self):
        if not self.is_processing:
            return  # Prevent duplicate processing_done calls
        self.reset_ui()
        self.status_label.configure(text="Status: Idle", text_color=LIGHT_GREEN)
        self.progress_var.set(1.0)
        messagebox.showinfo("Success", "Image processing completed successfully.")
    
    def processing_error(self, error_message):
        if not self.is_processing:
            return  # Prevent duplicate processing_error calls
        self.reset_ui()
        self.status_label.configure(text="Status: Error", text_color="red")
        self.progress_var.set(1.0)
        messagebox.showerror("Error", f"An error occurred: {error_message}")
    
    def update_status(self, message: str, progress: float):
        # Truncate message if too long
        max_length = 50
        if len(message) > max_length:
            message = message[:max_length - 3] + "..."
        # Update status label
        if message.startswith("Error:"):
            self.status_label.configure(text=message, text_color="red")
        elif message.startswith("Processing cancelled"):
            self.status_label.configure(text=message, text_color="orange")
        else:
            self.status_label.configure(text=message, text_color=LIGHT_BLUE)
        self.progress_var.set(progress)
        self.progress_bar.update()
