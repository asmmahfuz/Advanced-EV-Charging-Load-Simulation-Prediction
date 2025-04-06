from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import pandas as pd
import numpy as np
import os
import time # To measure performance
import traceback # For detailed error printing

# --- Import Simulation Engine ---
# Ensure simulation_engine.py is in the same directory or accessible via PYTHONPATH
try:
    from simulation_engine import load_and_preprocess_data, run_detailed_simulation
    SIMULATION_ENGINE_AVAILABLE = True
    print("Successfully imported simulation_engine.")
except ImportError as e:
    print(f"ERROR: Could not import from simulation_engine.py: {e}")
    print("Detailed simulation outputs will not be available.")
    SIMULATION_ENGINE_AVAILABLE = False
    # Define dummy functions if import fails to avoid NameError later
    def load_and_preprocess_data(*args, **kwargs):
        print("Warning: Using dummy load_and_preprocess_data.")
        return None
    def run_detailed_simulation(*args, **kwargs):
        print("Warning: Using dummy run_detailed_simulation.")
        return {} # Return empty dict

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates')

# --- Configuration & Model/Data Loading ---
MODEL_PATH = 'ev_load_predictor.joblib'
SCALER_PATH = 'scaler.joblib'
EV_DATA_FILE = "EVs_cases_1800_EVs_for_33_Bus_MV_distribution_network.xlsx" # Path to your Excel file

# Define expected features for ML model (MUST match training)
EXPECTED_FEATURES = [
    'evPercentage', 'chargingLogicCode', 'chargeDelay', 'peakStart', 'peakEnd',
    'peakReduction', 'socTarget', 'globalChargeLimit', 'scheduleShift',
    'socAdjustment', 'chargingEfficiency'
]
LOGIC_MAP_APP = {'immediate': 0, 'delayed': 1, 'offpeak': 2, 'soc_target': 3}

# --- Load ML Model, Scaler, and Base EV Data at Startup ---
ml_model = None
scaler = None
base_ev_data_global = None
startup_success = True # Flag to track if everything loaded

print("\n--- Application Startup Sequence ---")
try:
    # 1. Load ML Model
    print(f"Attempting to load ML model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"ML Model file not found at {MODEL_PATH}.")
    ml_model = joblib.load(MODEL_PATH)
    print("✅ ML model loaded successfully.")
    app.config['ML_MODEL'] = ml_model

    # 2. Load Scaler
    print(f"Attempting to load feature scaler from: {SCALER_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}.")
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded successfully.")
    app.config['SCALER'] = scaler

    # 3. Load Base EV Data (only if engine is available)
    if SIMULATION_ENGINE_AVAILABLE:
        print(f"Attempting to load base EV data from: {EV_DATA_FILE}")
        if not os.path.exists(EV_DATA_FILE):
             raise FileNotFoundError(f"Base EV data file not found at {EV_DATA_FILE}.")
        start_load_time = time.time()
        # Use try-except specifically for data loading issues
        try:
            base_ev_data_global = load_and_preprocess_data(filepath=EV_DATA_FILE)
            load_time = time.time() - start_load_time
            if base_ev_data_global is not None and not base_ev_data_global.empty:
                 print(f"✅ Base EV data loaded successfully ({base_ev_data_global.shape[0]} EVs) in {load_time:.2f} seconds.")
                 app.config['BASE_EV_DATA'] = base_ev_data_global # Store loaded data
                 app.config['SIMULATION_AVAILABLE'] = True # Confirm simulation is possible
            else:
                 # This case indicates load_and_preprocess returned None or empty df
                 print("⚠️ WARNING: Base EV data loaded as None or empty from simulation_engine. Detailed simulation disabled.")
                 app.config['BASE_EV_DATA'] = None
                 app.config['SIMULATION_AVAILABLE'] = False
                 # Don't mark startup as failed, but simulation won't run
        except Exception as data_load_error:
            print(f"❌ ERROR loading or preprocessing base EV data: {data_load_error}")
            traceback.print_exc()
            print("⚠️ WARNING: Detailed simulation disabled due to data loading failure.")
            app.config['BASE_EV_DATA'] = None
            app.config['SIMULATION_AVAILABLE'] = False
            # Don't mark startup as failed, but simulation won't run
    else:
        print("ℹ️ Skipping base EV data load as simulation engine is unavailable.")
        app.config['BASE_EV_DATA'] = None
        app.config['SIMULATION_AVAILABLE'] = False

    # Store other config items
    app.config['EXPECTED_FEATURES'] = EXPECTED_FEATURES
    app.config['LOGIC_MAP'] = LOGIC_MAP_APP
    print("--- Startup Sequence Complete ---")

except FileNotFoundError as fnf_error:
    print(f"❌ FATAL STARTUP ERROR: {fnf_error}")
    print("--- Application may not function correctly. ---")
    startup_success = False
except Exception as e:
    print(f"❌ FATAL STARTUP ERROR loading files: {e}")
    traceback.print_exc()
    print("--- Application may not function correctly. ---")
    startup_success = False


# --- Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    # Pass startup status to the template (optional)
    return render_template('index.html', startup_ok=startup_success, sim_available=app.config.get('SIMULATION_AVAILABLE', False))


@app.route('/simulate', methods=['POST'])
def simulate_ev_load_api():
    """
    API endpoint to predict EV load using ML model AND run detailed simulation.
    Handles errors from both steps distinctly.
    """
    start_request_time = time.time()
    results_dict = {}
    ml_success = False
    sim_success = False

    # --- Basic Startup Check ---
    # Check if essential ML components loaded, even if detailed sim failed
    if app.config.get('ML_MODEL') is None or app.config.get('SCALER') is None:
        print("Error: ML Model or Scaler not available (failed during startup).")
        return jsonify({'error': 'ML Model or Scaler not loaded on server startup. Cannot perform prediction.'}), 500

    # Determine if we should attempt detailed simulation
    run_detailed = app.config.get('SIMULATION_AVAILABLE', False) and app.config.get('BASE_EV_DATA') is not None
    if not run_detailed:
        print("Info: Detailed simulation will be skipped (Engine unavailable or Base EV Data failed to load).")


    # --- 1. Get Input Data ---
    try:
        params_raw = request.get_json()
        if not params_raw:
            return jsonify({'error': 'No input data provided.'}), 400
        print(f"Received simulation request with params: {params_raw}") # Log received params
    except Exception as e:
        print(f"Error parsing request JSON: {e}")
        return jsonify({'error': f'Invalid JSON format: {e}'}), 400

    # --- 2. ML Model Prediction ---
    try:
        print("Starting ML prediction...")
        # Prepare feature vector - use .get with defaults for robustness
        features = {}
        logic_str = params_raw.get('chargingLogic', 'immediate')
        logic_code = app.config['LOGIC_MAP'].get(logic_str)
        if logic_code is None:
            raise ValueError(f'Invalid chargingLogic received: {logic_str}')

        features['chargingLogicCode'] = logic_code
        features['evPercentage'] = float(params_raw.get('evPercentage', 100.0))
        # Apply conditional logic for parameters based on chargingLogic
        features['chargeDelay'] = int(params_raw.get('chargeDelay', 0)) if logic_str == 'delayed' else 0
        features['peakStart'] = int(params_raw.get('peakStart', 17)) if logic_str == 'offpeak' else 0 # Use defaults if logic matches but param missing
        features['peakEnd'] = int(params_raw.get('peakEnd', 20)) if logic_str == 'offpeak' else 0
        features['peakReduction'] = int(params_raw.get('peakReduction', 50)) if logic_str == 'offpeak' else 0
        features['socTarget'] = int(params_raw.get('socTarget', 80)) if logic_str == 'soc_target' else 0
        features['globalChargeLimit'] = float(params_raw.get('globalChargeLimit', 22.0))
        features['scheduleShift'] = int(params_raw.get('scheduleShift', 0))
        features['socAdjustment'] = int(params_raw.get('socAdjustment', 0))
        # Use the efficiency *parameter* for the ML model, as that's what it was trained on
        features['chargingEfficiency'] = int(params_raw.get('chargingEfficiency', 90))

        # Ensure features are in the correct order
        input_df = pd.DataFrame([features])[app.config['EXPECTED_FEATURES']]

        # Scale features
        input_scaled = app.config['SCALER'].transform(input_df)

        # Predict
        ml_predict_start = time.time()
        predicted_load_mw = app.config['ML_MODEL'].predict(input_scaled)[0] # [0] as predict returns a 2D array
        ml_predict_time = time.time() - ml_predict_start
        predicted_load_mw = np.maximum(0, predicted_load_mw) # Ensure non-negative predictions
        ml_hourly_load = predicted_load_mw.tolist()

        # Calculate ML-derived metrics
        ml_peak_load = float(np.max(ml_hourly_load)) if ml_hourly_load else 0.0
        ml_peak_time = int(np.argmax(ml_hourly_load)) if ml_peak_load > 1e-6 else 0 # Avoid argmax on all zeros
        ml_total_energy = float(np.sum(ml_hourly_load)) if ml_hourly_load else 0.0
        ml_avg_load = ml_total_energy / 24.0 if ml_hourly_load else 0.0
        ml_load_factor = (ml_avg_load / ml_peak_load * 100.0) if ml_peak_load > 1e-6 else 0.0
        ml_ldc = sorted(ml_hourly_load, reverse=True) # Load Duration Curve

        # Add ML results to the main dictionary
        results_dict['ml_hourlyLoad'] = ml_hourly_load
        results_dict['ml_peakLoad'] = ml_peak_load
        results_dict['ml_peakTime'] = ml_peak_time
        results_dict['ml_totalEnergy'] = ml_total_energy
        results_dict['ml_loadFactor'] = ml_load_factor
        results_dict['ml_ldc'] = ml_ldc
        results_dict['ml_predict_time'] = round(ml_predict_time, 3)
        ml_success = True
        print(f"✅ ML prediction successful ({ml_predict_time:.3f}s). Peak: {ml_peak_load:.2f} MW")

    except Exception as e:
        print(f"❌ Error during ML prediction step: {e}")
        traceback.print_exc()
        results_dict['error_ml'] = f"ML Prediction Failed: {str(e)}"
        # Provide empty ML results if prediction failed
        results_dict['ml_hourlyLoad'] = [0.0] * 24
        results_dict['ml_peakLoad'] = 0.0
        results_dict['ml_peakTime'] = 0
        results_dict['ml_totalEnergy'] = 0.0
        results_dict['ml_loadFactor'] = 0.0
        results_dict['ml_ldc'] = [0.0] * 24
        results_dict['ml_predict_time'] = 0.0
        # Continue to detailed sim if possible

    # --- 3. Detailed Simulation Run (Conditional) ---
    detailed_sim_time = 0
    if run_detailed:
        try:
            print("Starting detailed simulation...")
            detailed_sim_start = time.time()
            # Use the *raw* params from the request for the detailed sim function
            # The detailed sim function will use efficiency from the loaded data, ignoring the 'chargingEfficiency' param here
            sim_results_detailed = run_detailed_simulation(params_raw, app.config['BASE_EV_DATA'])
            detailed_sim_time = time.time() - detailed_sim_start
            print(f"✅ Detailed simulation completed ({detailed_sim_time:.2f}s). Peak: {sim_results_detailed.get('peakLoad', 0.0):.2f} MW")

            # Add detailed simulation results to the dictionary, using .get with defaults
            results_dict['sim_hourlyLoad'] = sim_results_detailed.get('hourlyLoad', [0.0]*24)
            results_dict['sim_peakLoad'] = sim_results_detailed.get('peakLoad', 0.0)
            results_dict['sim_totalEnergy'] = sim_results_detailed.get('totalEnergy', 0.0)
            results_dict['sim_concurrentEVs'] = sim_results_detailed.get('concurrentEVs', [0]*24)
            results_dict['sim_finalSOCs'] = sim_results_detailed.get('finalSOCs', [])
            results_dict['sim_avgFinalSoc'] = sim_results_detailed.get('avgFinalSoc', 0.0)
            results_dict['sim_busLoads'] = sim_results_detailed.get('busLoads', {}) # Hourly loads per bus
            results_dict['sim_numEVs'] = sim_results_detailed.get('numEVsSimulated', 0)
            results_dict['sim_detailed_time'] = round(detailed_sim_time, 2)
            sim_success = True

        except Exception as e:
            print(f"❌ Error during detailed simulation step: {e}")
            traceback.print_exc()
            results_dict['error_sim'] = f"Detailed Simulation Failed: {str(e)}"
            sim_success = False
            # Provide empty sim results if it failed
            results_dict['sim_hourlyLoad'] = [0.0]*24
            results_dict['sim_peakLoad'] = 0.0
            results_dict['sim_totalEnergy'] = 0.0
            results_dict['sim_concurrentEVs'] = [0]*24
            results_dict['sim_finalSOCs'] = []
            results_dict['sim_avgFinalSoc'] = 0.0
            results_dict['sim_busLoads'] = {}
            results_dict['sim_numEVs'] = 'Error'
            results_dict['sim_detailed_time'] = round(detailed_sim_time, 2) # Record time even if error occurred mid-way

    else:
        # If detailed sim was skipped, provide placeholder values
        print("Info: Populating simulation results with placeholders as detailed sim was skipped.")
        # Use ML results as fallback where appropriate, otherwise use zeros/empty
        results_dict['sim_hourlyLoad'] = results_dict.get('ml_hourlyLoad', [0.0]*24) # Use ML load if available
        results_dict['sim_peakLoad'] = results_dict.get('ml_peakLoad', 0.0)
        results_dict['sim_totalEnergy'] = results_dict.get('ml_totalEnergy', 0.0)
        results_dict['sim_concurrentEVs'] = [0]*24
        results_dict['sim_finalSOCs'] = []
        results_dict['sim_avgFinalSoc'] = 0.0
        results_dict['sim_busLoads'] = {}
        results_dict['sim_numEVs'] = 'N/A (Skipped)'
        results_dict['sim_detailed_time'] = 0.0
        # Mark sim as 'not run' rather than failed
        sim_success = False # Or a different status like 'skipped' could be used


    # --- 4. Post-Process Simulation Bus Loads (if sim ran successfully) ---
    top_bus_peaks = []
    if sim_success and 'sim_busLoads' in results_dict and results_dict['sim_busLoads']:
        try:
            bus_peaks = []
            for bus_id, hourly_loads_mw in results_dict['sim_busLoads'].items():
                # Ensure bus_id is treated as integer, handle potential non-numeric keys if necessary
                try:
                    bus_id_int = int(bus_id)
                except ValueError:
                    print(f"Warning: Non-integer bus ID found in busLoads: {bus_id}. Skipping.")
                    continue

                peak_load_mw = np.max(hourly_loads_mw) if hourly_loads_mw else 0.0
                bus_peaks.append({'busId': bus_id_int, 'peakLoadKw': peak_load_mw * 1000.0}) # Convert to kW

            # Sort by peak load descending and take top 5
            top_bus_peaks = sorted(bus_peaks, key=lambda x: x['peakLoadKw'], reverse=True)[:5]
        except Exception as e:
            print(f"Error processing bus loads: {e}")
            traceback.print_exc()
            results_dict['error_bus'] = "Failed to process bus peak loads."
            # top_bus_peaks remains empty or partially filled
    results_dict['sim_topBusPeaks'] = top_bus_peaks


    # --- 5. Finalize Response ---
    total_request_time = time.time() - start_request_time
    results_dict['total_request_time'] = round(total_request_time, 2)
    print(f"--- Request processing complete ({total_request_time:.2f}s) ---")

    # Determine overall status code
    status_code = 200 if (ml_success or sim_success) else 500 # OK if at least one part succeeded

    return jsonify(results_dict), status_code


# --- Main Execution ---
if __name__ == '__main__':
    print("\n--- Checking System Status Before Starting Server ---")
    if not startup_success:
         print("🔴 WARNING: Server starting with FATAL errors during initialization.")
         print("   ML predictions or detailed simulations may fail.")
    elif ml_model is None or scaler is None:
         print("🟡 WARNING: ML Model or Scaler failed to load. ML predictions will fail.")
    elif not app.config.get('SIMULATION_AVAILABLE', False):
         print("🟡 WARNING: Detailed simulation is disabled (engine unavailable or base data loading failed).")
    else:
        print("🟢 System status OK. ML model, scaler, and detailed simulation data loaded.")

    print("--- Starting Flask server ---")
    # Use host='0.0.0.0' to make accessible on network
    # debug=True enables auto-reloading and detailed error pages (disable in production)
    app.run(host='0.0.0.0', port=5000, debug=True)