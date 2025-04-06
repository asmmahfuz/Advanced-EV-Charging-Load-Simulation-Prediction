# Advanced EV Charging Load Simulation & Prediction

This project simulates aggregated Electric Vehicle (EV) charging load profiles based on user-defined parameters. It uniquely combines a fast Machine Learning (ML) prediction for the primary load profile with a detailed physics-based simulation for richer outputs like State of Charge (SOC) and concurrency.

The project was built from the ground up, starting with a publicly available dataset.

## Project Workflow & Development Steps

The development followed these key stages:

1.  **Data Acquisition:** The initial dataset (`EVs_cases_1800_EVs_for_33_Bus_MV_distribution_network.xlsx`) was obtained from the IEEE PES Working Group on Modern Heuristic Optimization public datasets: [IEEE Data Sets Link](https://site.ieee.org/pes-iss/data-sets/#elec). This dataset contains detailed characteristics and 24-hour connection schedules for 1800 EVs across 33 buses.

2.  **Data Exploration & Preprocessing (`simulation_engine.py`):**
    *   The raw Excel data presented a challenge due to its "long" format (16 rows per EV).
    *   The `load_and_preprocess_data` function was developed to parse this specific format, handle data cleaning (e.g., typos like "Minimun Charge"), and pivot the data into a "wide" format (one row per EV).
    *   Derived features like initial SOC (%) and the first connection hour were calculated.
    *   This preprocessing step resulted in a clean Pandas DataFrame suitable for simulation.

3.  **Detailed Simulation Engine Development (`simulation_engine.py`):**
    *   The `run_detailed_simulation` function was created to perform an hour-by-hour simulation based on the preprocessed data.
    *   It takes simulation parameters (charging logic, timing, limits, etc.) as input.
    *   It iterates through each EV for each hour, checking connection status, applying the selected charging logic, respecting physical constraints (capacity, max charge rate), and calculating energy drawn considering efficiency.
    *   It aggregates results like hourly load (MW), concurrent charging EVs, final SOC per EV, and load per bus.

4.  **Training Data Generation (`generate_data.py`):**
    *   To train an ML model for *fast* load prediction, a large dataset mapping input parameters to output load profiles was needed.
    *   `generate_data.py` was created to automate this. It repeatedly calls `run_detailed_simulation` (from `simulation_engine.py`) thousands of times with *randomized* input parameters (EV percentage, charging logic, limits, etc.).
    *   The 24-hour aggregated load profile output from *each* detailed simulation run, along with the corresponding input parameters, was saved as one row in `training_data.csv`. *(Note: This step was time-consuming)*.

5.  **Machine Learning Model Training (`train_model.py`):**
    *   `train_model.py` loads the generated `training_data.csv`.
    *   It separates input features (simulation parameters, numerically encoded) from the target variables (the 24 hourly load values).
    *   It uses `StandardScaler` from Scikit-learn to scale the input features (saving the scaler as `scaler.joblib`).
    *   It trains a `MultiOutputRegressor` wrapping a `RandomForestRegressor` to predict all 24 hourly load values simultaneously based on the scaled input features.
    *   The trained model was saved as `ev_load_predictor.joblib`.

6.  **Backend Development (`app.py`):**
    *   A Flask web server was created to host the application.
    *   On startup, `app.py` loads the pre-trained ML model (`ev_load_predictor.joblib`), the scaler (`scaler.joblib`), and the *preprocessed* base EV data (using `simulation_engine.load_and_preprocess_data` from the original Excel file). Preloading optimizes request handling time.
    *   It defines the `/simulate` API endpoint which:
        *   Receives user parameters via a POST request.
        *   Performs the ML prediction using the loaded model/scaler.
        *   Runs the detailed simulation using `simulation_engine.run_detailed_simulation` and the preloaded data.
        *   Combines results from both ML and detailed simulation into a single JSON response.

7.  **Frontend Development (`templates/index.html`):**
    *   A single-page web interface was built using HTML, CSS, and JavaScript.
    *   It provides interactive controls for users to set simulation parameters.
    *   It uses JavaScript (`fetch`) to communicate with the Flask backend's `/simulate` endpoint.
    *   It utilizes Chart.js to visualize the results (load profiles, SOC distribution, etc.).
    *   It includes UI enhancements like informational popups, theme switching (Dark/Light), and dynamic text interpretation of results.

## Project Structure

.
├── venv/ # Virtual environment files (auto-generated)
├── templates/
│ └── index.html # Main frontend HTML, CSS, and JavaScript UI
├── app.py # Flask backend server, API endpoint, and orchestration logic
├── simulation_engine.py # Contains data loading/preprocessing AND detailed simulation functions
├── generate_data.py # [Offline Use] Script to generate training_data.csv using simulation_engine
├── train_model.py # [Offline Use] Script to train ML model from training_data.csv
├── EVs_cases_1800_EVs_...xlsx # Source dataset from IEEE
├── ev_load_predictor.joblib # [Generated Artifact] Saved pre-trained ML model
├── scaler.joblib # [Generated Artifact] Saved feature scaler for ML model
├── training_data.csv # [Generated Artifact] Data used for ML training
├── requirements.txt # Python package dependencies for running the app
└── README.md # This file


## Prerequisites to Run the Application

*   **Python:** Version 3.8 or newer recommended. (`pip` should be included).
*   **Git:** For cloning the repository.

## Setup and Installation to Run the Application

*(These steps assume you only want to run the final application, not regenerate data or retrain the model)*

1.  **Clone the Repository:**
    ```bash
    git clone <your-github-repository-url>
    cd <repository-folder-name>
    ```

2.  **Create and Activate a Virtual Environment:**
    *   Create: `python -m venv venv` *(Use `python3` if needed)*
    *   Activate:
        *   Windows: `.\venv\Scripts\activate`
        *   macOS/Linux: `source venv/bin/activate`
    *   *(Confirm `(venv)` appears in your prompt)*

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This installs Flask, Pandas, NumPy, Scikit-learn, Joblib, Openpyxl)*

## Running the Application

1.  **Ensure Virtual Environment is Active:** Check for `(venv)` in your terminal prompt.
2.  **Run the Flask Server:** From the project's root directory:
    ```bash
    python app.py
    ```
3.  **Wait for Server Startup:** Look for output indicating the server is running, typically on `http://127.0.0.1:5000`.
4.  **Access in Browser:** Open your web browser and go to `http://127.0.0.1:5000/`.

## Stopping the Application

*   Press `Ctrl + C` in the terminal where the server is running.

## Note on Re-generating Data / Re-training Model

The files `generate_data.py` and `train_model.py` are included to show the development process. **Running them is NOT required** to use the web application, as the necessary artifacts (`training_data.csv`, `ev_load_predictor.joblib`, `scaler.joblib`) are already provided (or assumed to be in the repo).

If you were to modify the core simulation logic in `simulation_engine.py` and wanted to update the ML model, the sequence would be:

1.  Modify `simulation_engine.py`.
2.  Run `python generate_data.py` (Warning: This can take many hours).
3.  Run `python train_model.py` to create new `.joblib` files.
4.  Restart the Flask application (`python app.py`).

