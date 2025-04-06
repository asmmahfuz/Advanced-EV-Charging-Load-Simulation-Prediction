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

* **`venv/`**: (Auto-generated) Virtual environment files.
* **`templates/`**:
    * `index.html`: Main frontend HTML, CSS, and JavaScript UI.
* **`app.py`**: Flask backend server, API endpoint, and orchestration logic.
* **`simulation_engine.py`**: Data loading/preprocessing AND detailed simulation functions.
* **`generate_data.py`**: [Offline Use] Script to generate `training_data.csv` using `simulation_engine`.
* **`train_model.py`**: [Offline Use] Script to train ML model from `training_data.csv`.
* **`EVs_cases_1800_EVs_...xlsx`**: Source dataset from IEEE.
* **`ev_load_predictor.joblib`**: [Generated Artifact] Saved pre-trained ML model.
* **`scaler.joblib`**: [Generated Artifact] Saved feature scaler for ML model.
* **`training_data.csv`**: [Generated Artifact] Data used for ML training.
* **`requirements.txt`**: Python package dependencies for running the app.
* **`README.md`**: This file.


## Setup and Installation to Run the Application

*(These steps assume you only want to run the final application, not regenerate data or retrain the model)*

**Prerequisites:**

*   **Python:** Version 3.8 or newer is recommended. Verify with `python --version` or `python3 --version` in your terminal.
*   **Git:** For cloning the repository.

**Choose Your Operating System and Follow the Corresponding Instructions:**

**A. Windows - Using Command Prompt**

1.  **Open Command Prompt:** Search for "Command Prompt" in the Windows Start Menu and open it.

2.  **Navigate to Your Desktop (or Desired Location):**
    ```bash
    cd Desktop
    ```

3.  **Clone the GitHub Repository:**
    ```bash
    git clone https://github.com/asmmahfuz/Advanced-EV-Charging-Load-Simulation-Prediction.git
    ```

4.  **Change Directory into the Project Folder:**
    ```bash
    cd Advanced-EV-Charging-Load-Simulation-Prediction
    ```

5.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
    *   You should see `(venv)` at the beginning of your command prompt.

6.  **Install Python Libraries Manually (One by One):**
    *   **Important:** Ensure `(venv)` is active. Run these commands *one at a time*, and wait for each to complete successfully before proceeding.
        ```bash
        pip install Flask
        pip install pandas
        pip install numpy
        pip install scikit-learn
        pip install joblib
        pip install openpyxl
        ```

7.  **Run the Application:**
    ```bash
    python app.py
    ```

8.  **Wait for Server Startup:** Look for output indicating the server is running, typically on `http://127.0.0.1:5000`.

9.  **Access in Browser:** Open your web browser and go to `http://127.0.0.1:5000/`.

10. **Stopping the Application:** Press `Ctrl + C` in the terminal where the server is running.


**B. Windows - Using PowerShell**

1.  **Open PowerShell:** Search for "PowerShell" in the Windows Start Menu and open it.

2.  **Navigate to Your Desktop (or Desired Location):**
    ```powershell
    cd Desktop
    ```

3.  **Clone the GitHub Repository:**
    ```powershell
    git clone https://github.com/asmmahfuz/Advanced-EV-Charging-Load-Simulation-Prediction.git
    ```

4.  **Change Directory into the Project Folder:**
    ```powershell
    cd Advanced-EV-Charging-Load-Simulation-Prediction
    ```

5.  **Create and Activate Virtual Environment:**
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```
    *   You should see `(venv)` at the beginning of your PowerShell prompt.

6.  **Install Python Libraries Manually (One by One):**
    *   **Important:** Ensure `(venv)` is active. Run these commands *one at a time*, and wait for each to complete successfully before proceeding.
        ```powershell
        pip install Flask
        pip install pandas
        pip install numpy
        pip install scikit-learn
        pip install joblib
        pip install openpyxl
        ```

7.  **Run the Application:**
    ```powershell
    python app.py
    ```

8.  **Wait for Server Startup:** Look for output indicating the server is running, typically on `http://127.0.0.1:5000`.

9.  **Access in Browser:** Open your web browser and go to `http://127.0.0.1:5000/`.

10. **Stopping the Application:** Press `Ctrl + C` in the terminal where the server is running.

**C. Using VS Code Integrated Terminal (Windows or macOS/Linux)**

1.  **Open Your Project Folder in VS Code:** Open the `Advanced-EV-Charging-Load-Simulation-Prediction` folder.

2.  **Open Integrated Terminal:** Go to `Terminal > New Terminal`. VS Code will open a terminal panel within VS Code, usually at the bottom.

3.  **Create and Activate Virtual Environment:**
    *   In the VS Code terminal:
        ```bash
        python -m venv venv
        ```
    *   **Activate the virtual environment:** The activation command depends on your default shell in VS Code (check the top-right corner of the terminal panel - it might say PowerShell, Command Prompt, Bash, Zsh, etc.). Use the appropriate command:
        *   **PowerShell:** `.\venv\Scripts\activate`
        *   **Command Prompt:** `venv\Scripts\activate`
        *   **Bash/Zsh:** `source venv/bin/activate`
    *   You should see `(venv)` at the beginning of the terminal prompt.
    *   **Important: Select Python Interpreter:** VS Code might prompt you to select a Python interpreter. If it does, choose the one that is inside your `venv` folder (it will likely be something like `./venv/Scripts/python` or `./venv/bin/python`). If not prompted, you can manually select it: Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`), type "Python: Select Interpreter", and choose the `venv` interpreter.

4.  **Install Python Libraries Manually (One by One):**
    *   **Important:** Ensure `(venv)` is active and the correct Python interpreter is selected in VS Code. Run these commands *one at a time*:
        ```bash
        pip install Flask
        pip install pandas
        pip install numpy
        pip install scikit-learn
        pip install joblib
        pip install openpyxl
        ```

5.  **Run the Application:**
    ```bash
    python app.py
    ```

6.  **Wait for Server Startup:** Look for output indicating the server is running, typically on `http://127.0.0.1:5000`.

7.  **Access in Browser:** Open your web browser and go to `http://127.0.0.1:5000/`.

8. **Stopping the Application:** Press `Ctrl + C` in the terminal where the server is running.

**D. macOS or Linux Terminal (Bash/Zsh)**

1.  **Open Terminal:** Open the Terminal application on your macOS or Linux system.

2.  **Navigate to Your Desktop (or Desired Location):**
    ```bash
    cd Desktop
    ```

3.  **Clone the GitHub Repository:**
    ```bash
    git clone https://github.com/asmmahfuz/Advanced-EV-Charging-Load-Simulation-Prediction.git
    ```

4.  **Change Directory into the Project Folder:**
    ```bash
    cd Advanced-EV-Charging-Load-Simulation-Prediction
    ```

5.  **Create and Activate Virtual Environment:**
    ```bash
    python3 -m venv venv  # Use python3 on macOS/Linux
    source venv/bin/activate
    ```
    *   You should see `(venv)` at the beginning of your terminal prompt.

6.  **Install Python Libraries Manually (One by One):**
    *   **Important:** Ensure `(venv)` is active. Run these commands *one at a time*:
        ```bash
        pip install Flask
        pip install pandas
        pip install numpy
        pip install scikit-learn
        pip install joblib
        pip install openpyxl
        ```

7.  **Run the Application:**
    ```bash
    python3 app.py  # Use python3 on macOS/Linux if that's your Python 3 command
    ```

8.  **Wait for Server Startup:** Look for output indicating the server is running, typically on `http://127.0.0.1:5000`.

9.  **Access in Browser:** Open your web browser and go to `http://127.0.0.1:5000/`.

10. **Stopping the Application:** Press `Ctrl + C` in the terminal where the server is running.

## Note on Re-generating Data / Re-training Model

The files `generate_data.py` and `train_model.py` are included to show the development process. **Running them is NOT required** to use the web application, as the necessary artifacts (`training_data.csv`, `ev_load_predictor.joblib`, `scaler.joblib`) are already provided (or assumed to be in the repo).

If you were to modify the core simulation logic in `simulation_engine.py` and wanted to update the ML model, the sequence would be:

1.  Modify `simulation_engine.py`.
2.  Run `python generate_data.py` (Warning: This can take many hours).
3.  Run `python train_model.py` to create new `.joblib` files.
4.  Restart the Flask application (`python app.py`).

