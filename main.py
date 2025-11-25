# Author: Jiahuan Chen
# Date: Nov 23, 2025
# Description: A GUI application for the Real Estate Price Predictor.
# This module builds the Tkinter interface, allowing users to load data, visualize it,
# train models, view evaluation metrics, and make predictions.
#
# UPDATE: Added 'Zip Code' field to prediction interface to support location-based pricing.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from model_utils import RealEstateModel
import os

class RealEstateApp:
    def __init__(self, root):
        """
        Initialize the Real Estate Application GUI.

        :param root: The root Tkinter window.
        :type root: tk.Tk
        """
        self.root = root
        self.root.title("Real Estate Price Predictor")
        self.root.geometry("1000x800")
        
        self.model_system = RealEstateModel()
        
        # Layout configuration
        self.create_scrollable_layout()
        
    def create_scrollable_layout(self):
        """Create the main scrollable container"""
        # 1. Create a Main Frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # 2. Create a Canvas
        self.canvas = tk.Canvas(main_frame)
        self.canvas.pack(side="left", fill="both", expand=True)

        # 3. Add Scrollbars
        y_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        y_scrollbar.pack(side="right", fill="y")
        
        x_scrollbar = ttk.Scrollbar(self.root, orient="horizontal", command=self.canvas.xview)
        x_scrollbar.pack(side="bottom", fill="x")

        # 4. Configure Canvas
        self.canvas.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # 5. Create the content frame inside canvas
        self.content_frame = ttk.Frame(self.canvas)
        
        # Add frame to canvas window
        self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        # Now create the widgets inside content_frame
        self.create_widgets(self.content_frame)

    def create_widgets(self, parent):
        """
        Create and arrange all GUI widgets inside the parent frame.

        :param parent: The parent widget (frame) to put content in.
        """
        # --- Left Panel: Controls ---
        control_frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Data Loading
        ttk.Label(control_frame, text="1. Data Management").pack(fill="x", pady=(0, 5))
        self.load_btn = ttk.Button(control_frame, text="Load Data (CSV)", command=self.load_data)
        self.load_btn.pack(fill="x", pady=2)
        self.status_lbl = ttk.Label(control_frame, text="No data loaded", foreground="gray", wraplength=200) # Added wraplength
        self.status_lbl.pack(fill="x", pady=2)
        
        # Visualization
        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(control_frame, text="2. Visualization").pack(fill="x", pady=(0, 5))
        self.viz_corr_btn = ttk.Button(control_frame, text="Show Correlation", command=self.show_correlation)
        self.viz_corr_btn.pack(fill="x", pady=2)
        self.viz_dist_btn = ttk.Button(control_frame, text="Show Price Dist.", command=self.show_distribution)
        self.viz_dist_btn.pack(fill="x", pady=2)
        
        # Model Training
        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(control_frame, text="3. Model Training").pack(fill="x", pady=(0, 5))
        self.model_var = tk.StringVar(value="Random Forest")
        ttk.OptionMenu(control_frame, self.model_var, "Random Forest", "Linear Regression", "Random Forest", "Gradient Boosting").pack(fill="x", pady=2)
        
        # Added Cross-Validation Checkbox
        self.cv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Use 5-Fold CV (Slower)", variable=self.cv_var).pack(fill="x", pady=2)
        
        self.train_btn = ttk.Button(control_frame, text="Train Model", command=self.train_model)
        self.train_btn.pack(fill="x", pady=2)
        
       

    def load_data(self):
        """
        Handle the data loading process via GUI.
        
        :return: None
        """
        # Prefer default file if exists, else ask
        default_file = "realtor-data.zip.csv"
        if os.path.exists(default_file):
            filepath = default_file
        else:
            filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
            
        if not filepath:
            return
            
        self.status_lbl.config(text="Loading...", foreground="blue")
        self.root.update()
        
        # Load full dataset (sample_size=None) for maximum accuracy
        msg = self.model_system.load_data(filepath, sample_size=None)
        self.status_lbl.config(text=msg[:30] + "...", foreground="green")
        messagebox.showinfo("Info", msg)

    def display_figure(self, fig):
        """
        Display a matplotlib figure on the Tkinter canvas.

        :param fig: The matplotlib figure to display.
        :type fig: matplotlib.figure.Figure
        :return: None
        """
        if fig is None:
            messagebox.showerror("Error", "No data to visualize.")
            return
            
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
            
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")

    def show_correlation(self):
        """
        Trigger the display of the correlation plot.

        :return: None
        """
        fig = self.model_system.get_correlation_plot()
        self.display_figure(fig)

    def show_distribution(self):
        """
        Trigger the display of the price distribution plot.

        :return: None
        """
        fig = self.model_system.get_distribution_plot()
        self.display_figure(fig)

    def train_model(self):
        """
        Handle model training via GUI interaction.

        :return: None
        """
        if self.model_system.data is None:
            messagebox.showwarning("Warning", "Load data first!")
            return
            
        self.status_lbl.config(text="Training...", foreground="blue")
        self.root.update()
        
        model_type = self.model_var.get()
        use_cv = self.cv_var.get()
        
        # Pass the CV checkbox value to the backend
        msg = self.model_system.train_model(model_type, use_cv=use_cv)
        metrics = self.model_system.evaluate_model()
        
        self.metrics_text.config(state="normal")
        self.metrics_text.delete(1.0, tk.END)
        for k, v in metrics.items():
            self.metrics_text.insert(tk.END, f"{k}: {v}\n")
        self.metrics_text.config(state="disabled")
        
        self.status_lbl.config(text="Model Trained", foreground="green")
        messagebox.showinfo("Info", msg)



