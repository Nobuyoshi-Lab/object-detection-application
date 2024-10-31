# main.py

from config import Config
from gui import App
import customtkinter as ctk
import multiprocessing

def main():
    config = Config()
    ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
    root = ctk.CTk()
    app = App(root, config)
    root.mainloop()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # Recommended for compatibility
    main()
