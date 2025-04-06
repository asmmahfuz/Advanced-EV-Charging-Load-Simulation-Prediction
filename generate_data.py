import pandas as pd
import numpy as np
import random
from tqdm import tqdm # Progress bar
import time
import os

# Import simulation functions from simulation_engine.py
from simulation_engine import load_and_preprocess_data, run_detailed_simulation

# --- Configuration ---
NUM_SIMULATIONS = 5000  # Number of data points to generate (adjust as needed)
OUTPUT_CSV = 'training_data.csv'
EV_DATA_FILE = "EVs_cases_1800_EVs_for_33_Bus_MV_distribution_network.xlsx"

# Define parameter ranges/options for random sampling
PARAM_SPACE = {
    'evPercentage': lambda: random.uniform(5, 100), # Float 5-100%
    # 'selectedBuses': lambda: 'all' if random.random() < 0.5 else random.sample(range(1, 34), k=random.randint(1, 10)), # Complex to encode, simplifying
    # 'selectedEvTypes': lambda: random.sample([1, 2, 3], k=random.randint(1, 3)), # Complex to encode, simplifying
    'chargingLogic': lambda: random.choice(['immediate', 'delayed', 'offpeak', 'soc_target']),
    'chargeDelay': lambda: random.randint(0, 8),          # Hours (relevant for 'delayed')
    'peakStart': lambda: random.randint(0, 23),         # Hour (relevant for 'offpeak')
    'peakEnd': lambda: random.randint(0, 23),           # Hour (relevant for 'offpeak') - Simple version, allow start >= end
    'peakReduction': lambda: random.randint(0, 100),       # % (relevant for 'offpeak')
    'socTarget': lambda: random.randint(50, 100),        # % (relevant for 'soc_target')
    'globalChargeLimit': lambda: random.uniform(3.0, 40.0), # kW (float)
    'scheduleShift': lambda: random.randint(-6, 6),       # Hours
    'socAdjustment': lambda: random.randint(-20, 20),     # %
    'chargingEfficiency': lambda: random.randint(80, 100),   # %
    # 'startTimeRandom': lambda: random.randint(0, 60) # Not used in simulation/ML features for now
}

# Define features to be saved in the CSV (must match keys used in prediction)
# We simplify: Bus/Type selection are handled *inside* simulation but not direct features for ML
# We encode chargingLogic
FEATURE_COLUMNS = [
    'evPercentage',
    'chargingLogicCode', # Encoded: immediate=0, delayed=1, offpeak=2, soc_target=3
    'chargeDelay',
    'peakStart',
    'peakEnd',
    'peakReduction',
    'socTarget',
    'globalChargeLimit',
    'scheduleShift',
    'socAdjustment',
    'chargingEfficiency'
]

# Define target columns (24 hourly load values)
TARGET_COLUMNS = [f'Load_{h}' for h in range(24)]

# Mapping for charging logic
LOGIC_MAP = {'immediate': 0, 'delayed': 1, 'offpeak': 2, 'soc_target': 3}

def generate_one_datapoint(base_ev_data):
    """Generates one row of training data."""
    params = {}
    # Sample parameters
    for key, sampler in PARAM_SPACE.items():
        params[key] = sampler()

    # Add derived/fixed parameters needed by simulation but maybe not direct features
    params['selectedBuses'] = 'all' # Simplify for ML feature set
    params['selectedEvTypes'] = [1, 2, 3] # Simplify for ML feature set

    # Run the detailed simulation
    sim_results = run_detailed_simulation(params, base_ev_data)

    # Prepare features for storing
    features = {}
    features['evPercentage'] = params['evPercentage']
    features['chargingLogicCode'] = LOGIC_MAP[params['chargingLogic']]
    features['chargeDelay'] = params['chargeDelay'] if params['chargingLogic'] == 'delayed' else 0 # Use default if logic inactive
    features['peakStart'] = params['peakStart'] if params['chargingLogic'] == 'offpeak' else 0
    features['peakEnd'] = params['peakEnd'] if params['chargingLogic'] == 'offpeak' else 0
    features['peakReduction'] = params['peakReduction'] if params['chargingLogic'] == 'offpeak' else 0
    features['socTarget'] = params['socTarget'] if params['chargingLogic'] == 'soc_target' else 0
    features['globalChargeLimit'] = params['globalChargeLimit']
    features['scheduleShift'] = params['scheduleShift']
    features['socAdjustment'] = params['socAdjustment']
    features['chargingEfficiency'] = params['chargingEfficiency']

    # Flatten features and targets into one dictionary for the row
    row_data = features
    hourly_load = sim_results['hourlyLoad']
    for h in range(24):
        row_data[f'Load_{h}'] = hourly_load[h]

    return row_data

# --- Main Script ---
if __name__ == "__main__":
    print("Starting Training Data Generation...")
    start_time = time.time()

    # Load base EV data ONCE
    print(f"Loading base EV data from '{EV_DATA_FILE}'...")
    try:
        base_ev_data = load_and_preprocess_data(EV_DATA_FILE)
        print(f"Base EV data loaded successfully. Shape: {base_ev_data.shape}")
    except Exception as e:
        print(f"FATAL ERROR loading base EV data: {e}")
        exit(1)

    all_data = []
    print(f"Generating {NUM_SIMULATIONS} simulation data points...")
    for _ in tqdm(range(NUM_SIMULATIONS), desc="Simulations"):
        try:
            data_point = generate_one_datapoint(base_ev_data)
            all_data.append(data_point)
        except Exception as e:
            print(f"\nError during simulation run: {e}. Skipping this point.")
            # import traceback # Optional: print full traceback for debugging
            # traceback.print_exc()

    # Convert list of dictionaries to DataFrame
    training_df = pd.DataFrame(all_data)

    # Ensure columns are in the desired order
    training_df = training_df[FEATURE_COLUMNS + TARGET_COLUMNS]

    print(f"\nGenerated {len(training_df)} valid data points.")

    # Save to CSV
    if not training_df.empty:
        print(f"Saving data to '{OUTPUT_CSV}'...")
        try:
            training_df.to_csv(OUTPUT_CSV, index=False)
            print("Data saved successfully.")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")
    else:
        print("No data generated, CSV file not saved.")


    end_time = time.time()
    print(f"Total generation time: {end_time - start_time:.2f} seconds.")