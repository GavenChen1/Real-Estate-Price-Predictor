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

    # def get_correlation_plot(self):
    #     """
    #     Generates a correlation heatmap figure.
    #     Includes the newly created 'zip_code_encoded' feature to show location correlation.

    #     :return: A matplotlib figure object containing the correlation heatmap.
    #     """
    #     if self.data is None:
    #         return None
        
    #     # Create a copy for visualization that includes the numeric encoded zip
    #     viz_data = self.data.copy()
        
    #     # Simple encoding for visualization if not yet trained
    #     if self.zip_encoding_map is None:
    #          # Calculate temporary means for visualization
    #          temp_map = viz_data.groupby('zip_code')[self.target].mean()
    #          viz_data['zip_code_encoded'] = viz_data['zip_code'].map(temp_map)
    #     else:
    #          viz_data['zip_code_encoded'] = viz_data['zip_code'].map(self.zip_encoding_map).fillna(self.global_mean_price)

    #     plt.figure(figsize=(8, 6))
    #     # correlate only numeric columns
    #     corr = viz_data[self.train_features + [self.target]].corr()
    #     sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    #     plt.title("Feature Correlation Matrix")
    #     plt.tight_layout()
    #     fig = plt.gcf()
    #     return fig

    # def get_distribution_plot(self):
    #     """
    #     Generates a price distribution plot.
    #     Filters out the top 1% of outliers to make the visualization more readable.

    #     :return: A matplotlib figure object containing the price distribution histogram.
    #     """
    #     if self.data is None:
    #         return None
            
    #     plt.figure(figsize=(8, 6))
        
    #     # Filter top 1% of outliers for better visualization
    #     threshold = self.data[self.target].quantile(0.99)
    #     plot_data = self.data[self.data[self.target] <= threshold]
        
    #     sns.histplot(plot_data[self.target], bins=50, kde=True)
    #     plt.title(f"Price Distribution (0 - 99th Percentile: < ${threshold:,.0f})")
    #     plt.xlabel("Price")
    #     plt.tight_layout()
    #     fig = plt.gcf()
    #     return fig

    # def train_model(self, model_type='Random Forest', use_cv=False):
    #     """
    #     Trains the machine learning model using Target Encoding for Zip Codes.
    #     Optionally performs K-Fold Cross-Validation.

    #     :param model_type: The type of model to train ('Random Forest', 'Linear Regression', or 'Gradient Boosting').
    #     :type model_type: str
    #     :param use_cv: Whether to use Cross-Validation (default: False).
    #     :type use_cv: bool
    #     :return: A status message indicating the training result.
    #     """
    #     if self.data is None:
    #         return "No data loaded."
            
    #     # --- Target Encoding Logic (Hierarchical) ---
    #     # 1. Calculate global mean price
    #     self.global_mean_price = self.data[self.target].mean()
        
    #     # 2. Calculate mean price per STATE (Backup Level 2)
    #     self.state_encoding_map = self.data.groupby('state')[self.target].mean()

    #     # 3. Calculate mean price per CITY (Backup Level 1)
    #     # Trust city if > 10 sales
    #     city_stats = self.data.groupby('city')[self.target].agg(['mean', 'count'])
    #     self.city_encoding_map = city_stats[city_stats['count'] >= 10]['mean']

    #     # 4. Calculate mean price per ZIP CODE (Primary Level)
    #     zip_stats = self.data.groupby('zip_code')[self.target].agg(['mean', 'count'])
    #     # Only trust zip codes with at least 20 sales
    #     self.zip_encoding_map = zip_stats[zip_stats['count'] >= 20]['mean']
        
    #     # 5. Apply the Hierarchical Mapping
    #     def get_location_score(row):
    #         # Try Zip Code first
    #         if row['zip_code'] in self.zip_encoding_map:
    #             return self.zip_encoding_map[row['zip_code']]
    #         # Try City second
    #         elif row['city'] in self.city_encoding_map:
    #             return self.city_encoding_map[row['city']]
    #         # Try State third
    #         elif row['state'] in self.state_encoding_map:
    #             return self.state_encoding_map[row['state']]
    #         # Fallback to Global
    #         else:
    #             return self.global_mean_price

    #     # Create the single encoded feature
    #     self.data['location_encoded'] = self.data.apply(get_location_score, axis=1)
    #     # -----------------------------

    #     X = self.data[self.train_features]
    #     # Apply Log Transformation to Target: log1p(x) = log(x + 1)
    #     y = np.log1p(self.data[self.target])

    #     # Initialize Model
    #     if model_type == 'Linear Regression':
    #         self.model = LinearRegression()
    #     elif model_type == 'Gradient Boosting':
    #         self.model = HistGradientBoostingRegressor(random_state=42)
    #     else:
    #         self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    #     # CROSS VALIDATION LOGIC
    #     if use_cv:
    #         # 5-Fold CV
    #         kf = KFold(n_splits=5, shuffle=True, random_state=42)
    #         # We use 'r2' scoring. Note: cross_val_score returns an array of scores.
    #         scores = cross_val_score(self.model, X, y, cv=kf, scoring='r2')
    #         self.cv_scores = {
    #             "Mean R2": f"{scores.mean():.4f}",
    #             "Std R2": f"{scores.std():.4f}",
    #             "Scores": [f"{s:.2f}" for s in scores]
    #         }
    #         msg_suffix = f" (CV R2: {scores.mean():.2f})"
    #     else:
    #         self.cv_scores = None
    #         msg_suffix = ""

    #     # Standard Train/Test Split for final model (needed for 'predict' function later)
    #     X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #     self.model.fit(X_train, y_train)
    #     # Predict on test set (Log scale)
    #     y_pred_log = self.model.predict(self.X_test)
        
    #     # Convert back to original scale for evaluation (exp(x) - 1)
    #     self.y_test = np.expm1(self.y_test)
    #     self.y_pred = np.expm1(y_pred_log)
        
    #     return f"{model_type} trained successfully{msg_suffix}."

    # def evaluate_model(self):
    #     """
    #     Returns evaluation metrics. Includes CV scores if available.

    #     :return: A dictionary containing MSE, RMSE, MAE, R2 Score, and CV info.
    #     """
    #     if self.model is None:
    #         return "Model not trained."
            
    #     metrics = {}
        
    #     # Standard Metrics on Test Set
    #     if self.y_test is not None:
    #         mse = mean_squared_error(self.y_test, self.y_pred)
    #         rmse = mse ** 0.5
    #         mae = mean_absolute_error(self.y_test, self.y_pred)
    #         r2 = r2_score(self.y_test, self.y_pred)
            
    #         metrics.update({
    #             "Test MSE": f"{mse:,.2f}",
    #             "Test RMSE": f"{rmse:,.2f}",
    #             "Test MAE": f"{mae:,.2f}",
    #             "Test R2": f"{r2:.4f}"
    #         })
            
    #     # Add CV Metrics if they exist
    #     if self.cv_scores:
    #         metrics["--- CV Results ---"] = ""
    #         metrics["CV Mean R2"] = self.cv_scores["Mean R2"]
    #         metrics["CV Std Dev"] = self.cv_scores["Std R2"]
        
    #     return metrics

    # def predict(self, bed, bath, acre_lot, house_size, zip_code, city="", state=""):
    #     """
    #     Predicts price for given inputs using hierarchical location encoding.

    #     :return: A tuple (formatted_price_string, location_score_string)
    #     """
    #     if self.model is None:
    #         return "Model not trained.", "N/A"
            
    #     # Handle the inputs
    #     zip_str = str(zip_code).replace('.0', '')
        
    #     # Hierarchical Lookup
    #     if zip_str in self.zip_encoding_map:
    #         encoded_val = self.zip_encoding_map[zip_str]
    #         loc_status = f"${encoded_val:,.0f} (Zip Avg)"
    #     elif city in self.city_encoding_map:
    #         encoded_val = self.city_encoding_map[city]
    #         loc_status = f"${encoded_val:,.0f} (City Avg)"
    #     elif state in self.state_encoding_map:
    #         encoded_val = self.state_encoding_map[state]
    #         loc_status = f"${encoded_val:,.0f} (State Avg)"
    #     else:
    #         encoded_val = self.global_mean_price
    #         loc_status = "Global Avg (Loc Unknown)"
            
    #     input_data = pd.DataFrame([[bed, bath, acre_lot, house_size, encoded_val]], columns=self.train_features)
        
    #     # Predict (returns log price)
    #     log_prediction = self.model.predict(input_data)[0]
    #     # Convert back to actual price
    #     prediction = np.expm1(log_prediction)
        
    #     return f"${prediction:,.2f}", loc_status
