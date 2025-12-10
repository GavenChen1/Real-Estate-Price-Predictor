# Real Estate Price Predictor

A machine learning application to predict house prices based on property features.

## Features
1.  **Load Data**: Load real estate data from CSV.
2.  **Visualization**: View feature correlations and price distributions.
3.  **Model Training**: Train Random Forest or Linear Regression models.
4.  **Evaluation**: View model performance metrics (MSE, R2, etc.).
5.  **Prediction**: Predict prices for custom property inputs.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the application:
    ```bash
    python main.py
    ```

## Usage
- Ensure `realtor-data.zip.csv` is in the project directory or select it manually using the "Load Data" button.
- Click "Train Model" to build the predictor.
- Use the input fields to predict prices for new properties.
- **Sample Size Control**: Enter a number in the "Sample Size" field to limit the dataset size (default: 50000). Enter `0` to load the full dataset. Note: The full dataset is quite large and may take several minutes to train with Random Forest, so using a smaller sample size is recommended for faster testing and development.

## GUI Notes
- **Scrollbar Usage**: The scrollbar can only be used by clicking and dragging it up or down. The mouse scroll wheel does not work with the scrollbar in this application.

