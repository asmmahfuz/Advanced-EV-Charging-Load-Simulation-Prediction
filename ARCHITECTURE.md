# Internal Architecture Notes: Advanced EV Charging Load Simulation

## 1. Overview

This project implements a web application using a standard **Frontend-Backend** architecture.

*   **Frontend:** A single-page application (`index.html`) built with HTML, CSS, and JavaScript (using Chart.js for visualizations). It handles user interaction, parameter gathering, sending requests to the backend, and displaying results.
*   **Backend:** A Python **Flask** application (`app.py`) serving the frontend and providing a RESTful API endpoint (`/simulate`). It orchestrates the core logic, involving both a pre-trained Machine Learning model and a detailed simulation engine.
*   **Core Logic:** Resides primarily in `app.py` and `simulation_engine.py`. The backend loads necessary data and models on startup for efficiency.

## 2. Components

### 2.1. Frontend (`templates/index.html`)

*   **Role:** User Interface (UI), Input Parameter Collection, API Interaction, Results Visualization.
*   **Technologies:** HTML5, CSS3, JavaScript (ES6+), Chart.js library, Font Awesome (icons).
*   **Key JavaScript Logic:**
    *   **DOM Manipulation:** Gets references to all input/output elements.
    *   **Event Handling (`setupEventListeners`):** Attaches listeners to sliders, buttons, dropdowns, checkboxes, radio buttons, theme selector, and info icons.
    *   **Parameter Gathering (`gatherParameters`):** Collects values from all input elements into a JSON object suitable for the backend API.
    *   **API Call (`handleRunSimulation`):** Uses the `fetch` API to send a POST request with parameters to the `/simulate` endpoint. Handles response (success/error) and triggers UI updates.
    *   **UI Update (`updateUI`):** Parses the JSON response from the backend and updates metric values, Chart.js chart data, and the bus peak table.
    *   **Chart Management (`initializeCharts`, `updateChartsTheme`):** Creates and updates Chart.js instances, applying theme-specific styling.
    *   **Fleet Stats (`updateFleetStatsDisplay`, `getSelectedEvData`, `updateCapacityChart`):** Client-side calculation using simplified/mock data to provide immediate feedback on fleet composition as filters change.
    *   **Info Modal (`showInfoModal`, `hideInfoModal`):** Manages the display of informational popups based on clicked icons and content stored in `infoContent`.
    *   **Theme Handling (`applyTheme`, `loadInitialTheme`, `handleThemeChange`):** Manages CSS variables and chart updates for Dark/Light themes, storing preference in `localStorage`.
    *   **Interpretation (`generateInterpretationText`):** Dynamically creates a textual summary of the results based on inputs and outputs.

### 2.2. Backend (`app.py`)

*   **Role:** Web Server, API Endpoint Provider, Orchestration Layer.
*   **Technologies:** Python 3, Flask.
*   **Startup Process:**
    1.  Imports necessary libraries and the `simulation_engine`.
    2.  Loads the pre-trained ML model (`ev_load_predictor.joblib`) via `joblib`.
    3.  Loads the feature scaler (`scaler.joblib`) via `joblib`.
    4.  Loads and preprocesses the base EV data from the Excel file using `simulation_engine.load_and_preprocess_data()`, storing it in `base_ev_data_global`.
    5.  Stores loaded objects and configuration (e.g., `EXPECTED_FEATURES`, `LOGIC_MAP`) in Flask's `app.config`.
    6.  Starts the Flask development server.
*   **Routes:**
    *   `/`: Serves the main `index.html` file using `render_template`.
    *   `/simulate` (POST): The core API endpoint (details below).

### 2.3. Simulation Engine (`simulation_engine.py`)

*   **Role:** Handles raw EV data processing and performs the detailed, step-by-step charging simulation.
*   **Key Functions:**
    *   **`load_and_preprocess_data(filepath, sheet_name)`:**
        *   Reads the specific multi-row format from the Excel sheet (`.xlsx`).
        *   Handles data cleaning (whitespace, typo "Minimun Charge").
        *   Pivots the data from the long format (16 rows/EV) to a wide format (1 row/EV) Pandas DataFrame.
        *   Calculates derived fields like `InitialSOC_pct`, `FirstConnectedHour`.
        *   Performs final cleaning, NaN handling, and filtering.
        *   Returns the processed DataFrame ready for simulation.
    *   **`run_detailed_simulation(params, base_ev_data)`:**
        *   Takes simulation parameters (`params` from the user request) and the pre-processed `base_ev_data` DataFrame.
        *   Filters the DataFrame based on `evPercentage`, `selectedBuses`, `selectedEvTypes`.
        *   Initializes EV states (`CurrentSOC_MWh`, `CurrentSOC_pct`) applying `socAdjustment`.
        *   Loops through 24 hours:
            *   For each EV, checks connection status (considering `scheduleShift`), charging logic (`immediate`, `delayed`, `offpeak`, `soc_target`), and physical limits (`MaxChargeRateMW`, `globalChargeLimit`, capacity, SOC).
            *   Calculates the energy added and power drawn for the hour, considering `ChargeEfficiency` (from the *data*, not the user parameter which is for ML).
            *   Updates the EV's SOC.
            *   Aggregates hourly total load, concurrent charging EVs, and load per bus.
        *   Calculates final metrics (peak load, total energy, average final SOC).
        *   Returns a dictionary containing detailed simulation results.

### 2.4. ML Model & Scaler (`*.joblib`)

*   **Role:** Provide a fast prediction of the 24-hour aggregated charging load profile.
*   **Model:** `ev_load_predictor.joblib` contains a Scikit-learn `MultiOutputRegressor(RandomForestRegressor())` model.
*   **Scaler:** `scaler.joblib` contains a Scikit-learn `StandardScaler` fitted on the features of the training data. It's used to scale input features before prediction.
*   **Training:** Trained offline using `train_model.py` on data generated by `generate_data.py` (which used `run_detailed_simulation`).
*   **Input Features:** Expects a specific set of scaled numerical features as defined in `EXPECTED_FEATURES` in `app.py` (e.g., `evPercentage`, `chargingLogicCode`, `globalChargeLimit`, etc.).

### 2.5. Data Files (`.xlsx`, `.csv`)

*   **`EVs_cases_1800_EVs_...xlsx`:** The primary source data containing characteristics and connection schedules for 1800 EVs in a specific long format.
*   **`training_data.csv`:** (Generated offline) Contains input parameter combinations and the corresponding 24-hour load profiles *output by the detailed simulation*. Used to train the ML model. Not needed for running the application.

### 2.6. Offline Scripts (`generate_data.py`, `train_model.py`)

*   **Role:** One-time setup scripts.
*   **`generate_data.py`:** Ran `run_detailed_simulation` repeatedly with randomized parameters to create `training_data.csv`.
*   **`train_model.py`:** Loaded `training_data.csv`, scaled features, trained the `MultiOutputRegressor` model, and saved the model and scaler to `.joblib` files.

## 3. Data Flow (Simulation Request)

1.  **User Interaction:** User adjusts parameters in the frontend UI (`index.html`).
2.  **Parameter Gathering:** User clicks "Run Simulation". JavaScript (`gatherParameters`) collects UI values into a JSON object.
3.  **API Request:** JavaScript (`handleRunSimulation`) sends a POST request to `/simulate` with the JSON parameter object in the request body.
4.  **Backend Processing (`app.py` / `/simulate`):**
    a.  Receives and validates the JSON parameters.
    b.  **ML Prediction:**
        i.  Prepares the feature vector based on received parameters (encodes logic, selects features).
        ii. Uses the loaded `scaler` to transform the features.
        iii. Uses the loaded `ml_model` to predict the 24-hour load (`ml_hourlyLoad`).
        iv. Calculates derived ML metrics (peak, energy, LF, LDC).
    c.  **Detailed Simulation:**
        i.  Calls `run_detailed_simulation` function from `simulation_engine.py`, passing the *raw user parameters* and the preloaded `base_ev_data_global`.
        ii. The function filters EVs, simulates hour-by-hour charging based on physics and logic, and returns detailed results (`sim_hourlyLoad`, `sim_concurrentEVs`, `sim_finalSOCs`, `sim_busLoads`, etc.).
    d.  **Response Assembly:** Combines results from both ML prediction (prefixed `ml_...`) and detailed simulation (prefixed `sim_...`) into a single JSON object. Calculates top bus peaks from simulation results. Adds timing information.
5.  **API Response:** Backend sends the combined JSON results back to the frontend.
6.  **Frontend Update:** JavaScript (`updateUI`) receives the JSON response.
    a.  Updates the metric boxes with `ml_...` and `sim_...` values.
    b.  Updates the Chart.js charts with relevant data (ML load profile, Sim concurrent EVs, Sim SOC distribution, ML LDC).
    c.  Populates the Top Bus Peaks table.
    d.  Calls `generateInterpretationText` to display the narrative summary.
    e.  Updates the status display.

## 4. Key Design Decisions

*   **Hybrid ML + Detailed Simulation:** The core design uses ML for near-instantaneous feedback on the primary load profile, satisfying the main requirement. The detailed simulation runs concurrently to provide richer, physics-based outputs (SOC, concurrency, bus loads) that the ML model wasn't trained to predict directly. This balances speed and detail.
*   **Preloading Data and Models:** The ML model, scaler, and the large pre-processed EV dataset (`base_ev_data_global`) are loaded *once* when the Flask app starts. This significantly speeds up individual `/simulate` requests, as data loading/preprocessing doesn't happen repeatedly.
*   **Flask Backend:** A lightweight and simple Python web framework suitable for serving the single-page frontend and providing the API endpoint.
*   **Client-Side Fleet Stats:** Calculating basic fleet overview stats (avg capacity/rate, type distribution) in the browser JavaScript provides instant feedback as the user changes filters, without needing a backend call for this specific preview.

## 5. API Endpoint: `/simulate`

*   **Method:** POST
*   **Path:** `/simulate`
*   **Request Body:** JSON object containing user parameters. Expected keys (match `gatherParameters` in JS):
    ```json
    {
        "evPercentage": 100,
        "selectedBuses": "all" | [1, 5, 10],
        "selectedEvTypes": [1, 2, 3],
        "chargingLogic": "immediate" | "delayed" | "offpeak" | "soc_target",
        "chargeDelay": 1, // Relevant if logic=delayed
        "peakStart": 17, // Relevant if logic=offpeak
        "peakEnd": 20, // Relevant if logic=offpeak
        "peakReduction": 50, // Relevant if logic=offpeak
        "socTarget": 80, // Relevant if logic=soc_target
        "globalChargeLimit": 22.0,
        "scheduleShift": 0,
        "socAdjustment": 0,
        "chargingEfficiency": 90, // Used for ML model feature
        "startTimeRandom": 5 // Included but currently unused by backend
    }
    ```
*   **Response Body (Success):** JSON object containing combined results. Key fields include:
    ```json
    {
        "ml_hourlyLoad": [ ... ], // 24 values (MW)
        "ml_peakLoad": 5.61,
        "ml_totalEnergy": 127.06,
        "ml_loadFactor": 94.4,
        "ml_peakTime": 23,
        "ml_ldc": [ ... ], // 24 sorted values (MW)
        "ml_predict_time": 0.05,
        "sim_hourlyLoad": [ ... ], // 24 values (MW)
        "sim_peakLoad": 5.70,
        "sim_totalEnergy": 128.37,
        "sim_concurrentEVs": [ ... ], // 24 integer values
        "sim_finalSOCs": [ ... ], // List of final SOC percentages for simulated EVs
        "sim_avgFinalSoc": 4.5,
        "sim_busLoads": { "1": [ ... ], "4": [ ... ], ... }, // Dict: busId -> 24 hourly loads (MW)
        "sim_numEVs": 1798,
        "sim_detailed_time": 10.17,
        "sim_topBusPeaks": [ { "busId": 4, "peakLoadKw": 241.00 }, ... ], // Top 5 bus peaks
        "total_request_time": 10.22,
        // Optional error fields if a step fails:
        // "error_ml": "Description of ML error",
        // "error_sim": "Description of Sim error",
        // "error_bus": "Description of bus processing error"
    }
    ```
*   **Response Body (Error):** JSON object with an `error` key, or potentially partial results with specific `error_ml`/`error_sim` keys. Status code might be 4xx or 5xx.
    ```json
    { "error": "Description of the error (e.g., Invalid JSON, Model not loaded)" }
    ```