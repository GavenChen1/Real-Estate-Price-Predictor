# Author: Jiahuan Chen
# Date: Nov 23, 2025
# Description: A machine learning application to predict house prices based on property features.
# This module handles data loading, preprocessing, model training (Random Forest/Linear Regression),
# and prediction logic. It requires pandas, scikit-learn, and matplotlib.


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io

class RealEstateModel:
    def __init__(self):
        self.data = None
        self.model = None
        # Added 'city' to features for Hierarchical Target Encoding
        self.features = ['bed', 'bath', 'acre_lot', 'house_size', 'zip_code', 'city', 'state']
        self.train_features = ['bed', 'bath', 'acre_lot', 'house_size', 'location_encoded']
        self.target = 'price'
        self.zip_encoding_map = None
        self.city_encoding_map = None # Avg price per city (Backup 1)
        self.state_encoding_map = None # Avg price per state (Backup 2)
        self.global_mean_price = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.cv_scores = None

    def load_data(self, filepath, sample_size=None):
        """
        Loads data from CSV, handles basic cleaning.
        
        :param filepath: Path to the CSV file containing real estate data.
        :type filepath: str
        :param sample_size: Number of rows to sample. If None, loads full dataset.
        :type sample_size: int or None
        :return: A status message indicating success or failure.
        """
        try:
            # STEP 1: Read the CSV file into a pandas DataFrame
            # We use pandas to handle potentially large datasets efficiently
            df = pd.read_csv(filepath)
            
            # Ensure zip_code is treated as string/categorical, not a number
            df['zip_code'] = df['zip_code'].astype(str).str.replace('.0', '', regex=False)
            
            # STEP 2: Remove rows with missing values (Data Cleaning)
            # We drop any row where 'bed', 'bath', 'acre_lot', 'house_size', or 'price' is NaN (null).
            # We also need zip_code to be present now.
            df = df.dropna(subset=self.features + [self.target])
            
            # STEP 3: Filter out invalid or illogical data (Outlier Removal)
            # Prices must be positive (greater than 0)
            df = df[df['price'] > 0]
            # House size must be positive (greater than 0 sqft)
            df = df[df['house_size'] > 0]
            
            # STEP 4: Downsample the dataset (Performance Optimization)
            # Only sample if sample_size is explicitly provided
            if sample_size is not None and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                
            self.data = df
            return f"Data loaded successfully. shape: {self.data.shape}"
        except Exception as e:
            return f"Error loading data: {str(e)}"

    def get_correlation_plot(self):
        """
        Generates a correlation heatmap figure.
        Includes the newly created 'zip_code_encoded' feature to show location correlation.

        :return: A matplotlib figure object containing the correlation heatmap.
        """
        if self.data is None:
            return None
        
        # Create a copy for visualization that includes the numeric encoded zip
        viz_data = self.data.copy()
        
        # Simple encoding for visualization if not yet trained
        if self.zip_encoding_map is None:
             # Calculate temporary means for visualization
             temp_map = viz_data.groupby('zip_code')[self.target].mean()
             viz_data['zip_code_encoded'] = viz_data['zip_code'].map(temp_map)
        else:
             viz_data['zip_code_encoded'] = viz_data['zip_code'].map(self.zip_encoding_map).fillna(self.global_mean_price)

        plt.figure(figsize=(8, 6))
        # correlate only numeric columns
        corr = viz_data[self.train_features + [self.target]].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        fig = plt.gcf()
        return fig

