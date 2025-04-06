import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import time
import os

# --- Configuration ---
TRAINING_DATA_CSV = 'training_data.csv'
MODEL_SAVE_PATH = 'ev_load_predictor.joblib'
SCALER_SAVE_PATH = 'scaler.joblib'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Define feature and target columns (MUST match generate_data.py)
FEATURE_COLUMNS = [
    'evPercentage', 'chargingLogicCode', 'chargeDelay', 'peakStart', 'peakEnd',
    'peakReduction', 'socTarget', 'globalChargeLimit', 'scheduleShift',
    'socAdjustment', 'chargingEfficiency'
]
TARGET_COLUMNS = [f'Load_{h}' for h in range(24)]

# --- Main Training Script ---
if __name__ == "__main__":
    print("Starting ML Model Training...")
    start_time = time.time()

    # 1. Load Data
    print(f"Loading training data from '{TRAINING_DATA_CSV}'...")
    if not os.path.exists(TRAINING_DATA_CSV):
        print(f"Error: Training data file '{TRAINING_DATA_CSV}' not found.")
        print("Please run generate_data.py first.")
        exit(1)

    try:
        df = pd.read_csv(TRAINING_DATA_CSV)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit(1)

    # Check for required columns
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    missing_targets = [col for col in TARGET_COLUMNS if col not in df.columns]
    if missing_features or missing_targets:
        print("Error: Missing required columns in CSV.")
        if missing_features: print(f"  Missing Features: {missing_features}")
        if missing_targets: print(f"  Missing Targets: {missing_targets}")
        exit(1)

    # 2. Separate Features (X) and Targets (y)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]
    print(f"Features shape: {X.shape}, Targets shape: {y.shape}")

    # 3. Split Data into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # 4. Preprocess Features (Scaling)
    print("Scaling numerical features...")
    # All features in FEATURE_COLUMNS are treated as numerical here
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    try:
        joblib.dump(scaler, SCALER_SAVE_PATH)
        print(f"Scaler saved to '{SCALER_SAVE_PATH}'")
    except Exception as e:
        print(f"Error saving scaler: {e}")

    # 5. Define and Train the Model
    print("Defining and training the MultiOutput RandomForest model...")
    # Configure the base estimator (Random Forest) - parameters can be tuned
    base_estimator = RandomForestRegressor(
        n_estimators=100,       # Number of trees
        random_state=RANDOM_STATE,
        n_jobs=-1,             # Use all available CPU cores
        max_depth=25,          # Limit tree depth (tuning parameter)
        min_samples_split=10,  # Min samples to split a node (tuning parameter)
        min_samples_leaf=5,   # Min samples at a leaf node (tuning parameter)
        # oob_score=True       # Can use Out-of-Bag score for validation during training
    )

    # Wrap with MultiOutputRegressor
    multi_target_model = MultiOutputRegressor(base_estimator)

    # Train the model
    training_start_time = time.time()
    multi_target_model.fit(X_train_scaled, y_train)
    training_end_time = time.time()
    print(f"Model training completed in {training_end_time - training_start_time:.2f} seconds.")

    # 6. Evaluate the Model
    print("Evaluating model on the test set...")
    y_pred = multi_target_model.predict(X_test_scaled)

    # Calculate metrics
    mae_per_hour = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    overall_mae = np.mean(mae_per_hour)
    r2_per_hour = r2_score(y_test, y_pred, multioutput='raw_values')
    overall_r2 = r2_score(y_test, y_pred) # Overall R^2

    print(f"\nModel Evaluation Metrics:")
    print(f"  Overall Mean Absolute Error (MAE): {overall_mae:.4f} MW")
    print(f"  Overall R-squared (R2): {overall_r2:.4f}")
    # print(f"  MAE per hour: {np.round(mae_per_hour, 4)}") # Optionally print detailed per-hour MAE

    # 7. Save the Trained Model
    print(f"Saving trained model to '{MODEL_SAVE_PATH}'...")
    try:
        joblib.dump(multi_target_model, MODEL_SAVE_PATH)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

    end_time = time.time()
    print(f"\nTotal training script time: {end_time - start_time:.2f} seconds.")