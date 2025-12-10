# Author: Zhihao Wu
# Date: 2025-12-9
# test_ames_model.py
# Use ames_for_app.csv to test RealEstateModel and write results to test_report.txt

from model_utils import RealEstateModel
import contextlib

DATA_FILE = "ames_for_app.csv"


def run_one_model(model: RealEstateModel, model_type: str, use_cv: bool = True):
    """
    Train and evaluate a single model type and print metrics.
    All prints will be redirected to the report file by the caller.
    """
    print("\n==============================")
    print(f"Testing model type: {model_type}")
    print("==============================")

    # Train the model
    msg = model.train_model(model_type=model_type, use_cv=use_cv)
    print("Train message:", msg)

    # Ensure the model object exists after training
if model.model is None:
    print("ERROR: Model object was not created correctly!")
else:
    print("Model object created successfully.")

    # Evaluate on the hold-out test set
    metrics = model.evaluate_model()
    print("\n--- Metrics on hold-out test set ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Print cross-validation results if available
    if model.cv_scores is not None:
        print("\n--- 5-Fold CV R^2 ---")
        for k, v in model.cv_scores.items():
            print(f"{k}: {v}")

    # Use one row from the dataset as a sample prediction
    sample_row = model.data.iloc[0]
    pred_price, loc_info = model.predict(
        bed=float(sample_row["bed"]),
        bath=float(sample_row["bath"]),
        acre_lot=float(sample_row["acre_lot"]),
        house_size=float(sample_row["house_size"]),
        zip_code=str(sample_row["zip_code"]),
        city=str(sample_row["city"]),
        state=str(sample_row["state"])
    )
    print("\n--- Sample prediction ---")
    print("Input sample:", sample_row.to_dict())
    print("Predicted price:", pred_price)
    print("Location info:", loc_info)


def main():
    """
    Load data and run tests for three different model types.
    """
    # Load prepared Ames data
    model = RealEstateModel()
    msg = model.load_data(DATA_FILE, sample_size=None)
    print("Load message:", msg)

    # Test three model types with 5-fold cross-validation
    for m in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
        run_one_model(model, model_type=m, use_cv=True)


if __name__ == "__main__":
    # Redirect all stdout from main() into test_report.txt
    with open("test_report.txt", "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            main()

    # This line prints to the console (not into the file) to confirm completion
    print("Test report has been generated: test_report.txt")

