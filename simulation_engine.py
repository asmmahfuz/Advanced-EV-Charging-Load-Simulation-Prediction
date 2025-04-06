# simulation_engine.py
# No changes needed based on the request. Keeping the existing robust version.
import pandas as pd
import numpy as np
import random
import os
import traceback # For detailed error printing

def load_and_preprocess_data(filepath="EVs_cases_1800_EVs_for_33_Bus_MV_distribution_network.xlsx", sheet_name='V2G'):
    """
    Loads and preprocesses the EV data from the Excel file, handling the
    specific long format (16 rows per EV). Transforms data into a
    wide format (one row per EV) suitable for simulation.
    Handles typo 'Minimun Charge' and addresses FutureWarning.
    """
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        # Load the sheet, skipping the initial rows and using no header initially.
        # Data starts at Excel row 4, which is index 3. So skip 3 rows.
        df = pd.read_excel(filepath, sheet_name=sheet_name, header=None, skiprows=3)
        print(f"Initial loaded shape (header=None, skiprows=3): {df.shape}")

    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")

    if df.empty:
        raise ValueError("Loaded DataFrame is empty after skipping rows.")

    # --- Assign Column Names based on the structure description ---
    num_cols = df.shape[1]
    if num_cols < 31:
         raise ValueError(f"Expected at least 31 columns (A-AE), but found {num_cols}")

    col_names = ['row_num_excel', 'VehicleType', 'ChEff_pct', 'DchEff_pct', 'FeatureName',
                 'EmptyColF', 'FeatureIndexG']
    hour_cols_raw = [f'H{i}' for i in range(1, 25)] # H1 to H24 correspond to columns 7 to 30
    col_names.extend(hour_cols_raw)

    # Assign names to the relevant columns (up to 31)
    df.columns = col_names[:num_cols]

    # Drop the completely empty column F if it exists
    if 'EmptyColF' in df.columns:
        df.drop(columns=['EmptyColF'], inplace=True)

    # --- Strip Whitespace from FeatureName column ---
    if 'FeatureName' in df.columns:
        # Ensure it's string type before stripping
        df['FeatureName'] = df['FeatureName'].astype(str).str.strip()
        print("Stripped whitespace from 'FeatureName' column.")
        # print("Unique Feature Names after stripping:", df['FeatureName'].unique()) # Keep commented unless debugging
    else:
        raise ValueError("'FeatureName' column not found after initial load.")


    # --- Generate EV ID ---
    # Each EV has 16 rows, integer division gives the ID
    rows_per_ev = 16
    df['EV_ID'] = df.index // rows_per_ev
    print(f"Generated EV_ID range: {df['EV_ID'].min()} to {df['EV_ID'].max()}")


    # --- Pivot Data from Long to Wide Format ---

    # 1. Extract static data (first row per EV)
    static_df = df.groupby('EV_ID').first()[['VehicleType', 'ChEff_pct', 'DchEff_pct']].copy()
    static_df['ChargeEfficiency'] = pd.to_numeric(static_df['ChEff_pct'], errors='coerce') / 100.0
    static_df['DischargeEfficiency'] = pd.to_numeric(static_df['DchEff_pct'], errors='coerce') / 100.0 # Keep for potential V2G
    static_df.drop(columns=['ChEff_pct', 'DchEff_pct'], inplace=True)
    static_df['VehicleType'] = pd.to_numeric(static_df['VehicleType'], errors='coerce') # Assuming numeric type 1, 2, 3

    # 2. Extract "static-like" features stored row-wise
    # Use the *actual* names found in the Excel file (including the typo)
    static_like_feature_names = ['Capacity', 'Max Charge', 'Max Discharge', 'Minimun Charge', 'Previous State', 'Bus'] # <-- Adjusted for typo
    static_like_data = df[df['FeatureName'].isin(static_like_feature_names)].copy()

    # Check if the misspelled feature was found
    found_static_features_after_strip = static_like_data['FeatureName'].unique()
    if 'Minimun Charge' not in found_static_features_after_strip:
        print("Warning: 'Minimun Charge' (with typo) feature STILL not found even after stripping whitespace. Check Excel spelling.")
    missing_static_features = set(static_like_feature_names) - set(found_static_features_after_strip)
    if missing_static_features:
        print(f"Warning: Did not find data rows for the following expected static features: {missing_static_features}")

    # Identify the correct column index for H1 (should be 7 after dropping 'EmptyColF')
    h1_col_index = df.columns.get_loc('H1')
    static_like_data['Value'] = pd.to_numeric(static_like_data.iloc[:, h1_col_index], errors='coerce')
    static_pivoted = static_like_data.pivot_table(index='EV_ID', columns='FeatureName', values='Value')

    print("Columns in static_pivoted BEFORE rename:", static_pivoted.columns.tolist())

    # Rename pivoted static columns
    # Map the *actual* Excel name (with typo) to the desired final name
    static_rename_map = {
        'Capacity': 'CapacityMWh',
        'Max Charge': 'MaxChargeRateMW',
        'Max Discharge': 'MaxDischargeRateMW',
        'Minimun Charge': 'MinChargeLimitMWh', # <-- Key is the typo
        'Previous State': 'PreviousStateMWh',
        'Bus': 'Bus'
    }
    # Only rename columns that actually exist in the pivoted table
    rename_map_existing = {k: v for k, v in static_rename_map.items() if k in static_pivoted.columns}
    static_pivoted.rename(columns=rename_map_existing, inplace=True)

    print("Columns in static_pivoted AFTER rename:", static_pivoted.columns.tolist())

    # Ensure Bus is integer
    if 'Bus' in static_pivoted.columns:
         static_pivoted['Bus'] = pd.to_numeric(static_pivoted['Bus'], errors='coerce').round().astype('Int64') # Use nullable Int

    # 3. Extract truly time-varying features (Connection Status)
    # Identify the correct column indices for H1 to H24
    hour_col_indices = [df.columns.get_loc(f'H{i}') for i in range(1, 25)]
    hour_cols_actual = df.columns[hour_col_indices].tolist() # Get the actual column names at these indices

    connect_data_long = df[df['FeatureName'] == 'Connect (1/0)'][['EV_ID'] + hour_cols_actual].copy()
    if connect_data_long.empty:
        print("Warning: No data found for 'Connect (1/0)' feature. Creating placeholder columns (all zeros).")
        # Create placeholder columns matching the static_df index
        connect_data = pd.DataFrame(index=static_df.index)
        for i in range(24): connect_data[f'Connect_{i}'] = 0
    else:
        connect_data = connect_data_long.set_index('EV_ID')
        # Rename H1..H24 (or whatever the actual columns are) to Connect_0..Connect_23
        connect_rename_map = {hour_cols_actual[i]: f'Connect_{i}' for i in range(24)}
        connect_data.rename(columns=connect_rename_map, inplace=True)
        # Convert to numeric/integer, filling potential NaNs with 0 (not connected)
        for col in connect_data.columns:
            connect_data[col] = pd.to_numeric(connect_data[col], errors='coerce').fillna(0).astype(int)

    # 4. Join all parts together using EV_ID index
    final_df = static_df.join(static_pivoted, on='EV_ID', how='left')
    final_df = final_df.join(connect_data, on='EV_ID', how='left')
    print(f"Shape after joining components: {final_df.shape}")
    print("Columns in final_df after joining:", final_df.columns.tolist())

    # --- 5. Post-processing and Cleaning on the Wide DataFrame ---

    # Calculate Initial SOC Percentage
    final_df['CapacityMWh'] = pd.to_numeric(final_df.get('CapacityMWh'), errors='coerce')
    final_df['PreviousStateMWh'] = pd.to_numeric(final_df.get('PreviousStateMWh'), errors='coerce')

    final_df['InitialSOC_pct'] = 0.0
    valid_capacity_mask = (final_df['CapacityMWh'] > 1e-6) & (final_df['CapacityMWh'].notna())
    final_df.loc[valid_capacity_mask, 'InitialSOC_pct'] = \
        (final_df.loc[valid_capacity_mask, 'PreviousStateMWh'].fillna(0) / final_df.loc[valid_capacity_mask, 'CapacityMWh']) * 100
    final_df['InitialSOC_pct'] = final_df['InitialSOC_pct'].clip(0, 100).fillna(0)

    # Convert other relevant columns to numeric
    final_df['MaxChargeRateMW'] = pd.to_numeric(final_df.get('MaxChargeRateMW'), errors='coerce')

    # Conditional handling of MinChargeLimitMWh
    if 'MinChargeLimitMWh' in final_df.columns:
        final_df['MinChargeLimitMWh'] = pd.to_numeric(final_df['MinChargeLimitMWh'], errors='coerce')
        print("Converted 'MinChargeLimitMWh' column to numeric.")
    else:
        print("Warning: 'MinChargeLimitMWh' column not found after join/rename. Creating it with NaN values.")
        final_df['MinChargeLimitMWh'] = np.nan # Create column if missing

    # Handle remaining NaNs and Filter - Address FutureWarning
    # Use direct assignment instead of inplace=True on potentially chained objects
    final_df['MaxChargeRateMW'] = final_df['MaxChargeRateMW'].fillna(0)
    # Use a reasonable default efficiency if missing
    final_df['ChargeEfficiency'] = final_df['ChargeEfficiency'].fillna(0.90)
    final_df['DischargeEfficiency'] = final_df['DischargeEfficiency'].fillna(0.90)
    print("Filled NaNs in MaxChargeRateMW and Efficiencies.")

    # Drop rows that are unusable for simulation
    initial_rows = len(final_df)
    # Check existence before using in dropna subset
    dropna_cols = [col for col in ['VehicleType', 'Bus', 'InitialSOC_pct', 'CapacityMWh'] if col in final_df.columns]
    if len(dropna_cols) < 4:
        print(f"Warning: Missing one or more critical columns for dropna check: {set(['VehicleType', 'Bus', 'InitialSOC_pct', 'CapacityMWh']) - set(dropna_cols)}")
    # Use .dropna() directly on the DataFrame
    final_df = final_df.dropna(subset=dropna_cols)


    # Also remove EVs with non-positive capacity or zero charge rate (cannot charge)
    # Ensure CapacityMWh and MaxChargeRateMW columns exist before filtering
    if 'CapacityMWh' in final_df.columns and 'MaxChargeRateMW' in final_df.columns:
         final_df = final_df[(final_df['CapacityMWh'] > 1e-6) & (final_df['MaxChargeRateMW'] > 1e-6)]
    else:
         print("Warning: 'CapacityMWh' or 'MaxChargeRateMW' missing, cannot apply >0 filter.")

    rows_dropped = initial_rows - len(final_df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to missing critical data or zero capacity/charge rate.")

    # Calculate FirstConnectedHour using the Connect_0..Connect_23 columns
    connect_cols_list = [f'Connect_{h}' for h in range(24)]
    # Ensure all connect columns exist before applying
    if all(col in final_df.columns for col in connect_cols_list):
        # Efficiently find the first '1' using idxmax along axis 1 after checking if any '1' exists
        connection_times = final_df[connect_cols_list]
        any_connection = connection_times.any(axis=1)
        # idxmax returns the *column name* of the first max value (which is '1' here)
        first_connect_col_name = connection_times.idxmax(axis=1)
        # Extract the hour number from the column name 'Connect_X'
        final_df['FirstConnectedHour'] = first_connect_col_name.str.extract(r'Connect_(\d+)').astype(int)
        # Set to -1 where no connection exists
        final_df.loc[~any_connection, 'FirstConnectedHour'] = -1
    else:
        print("Warning: Not all 'Connect_x' columns found. Setting 'FirstConnectedHour' to -1.")
        final_df['FirstConnectedHour'] = -1

    # Reset index AFTER filtering to get consecutive indices if needed
    final_df.reset_index(inplace=True, drop=True) # drop=True prevents old index becoming a column


    # Select and order final columns needed for the simulation engine
    required_sim_cols = [
        'EV_ID', 'VehicleType', 'ChargeEfficiency', 'PreviousStateMWh',
        'MaxChargeRateMW', 'CapacityMWh', 'Bus', 'MinChargeLimitMWh',
        'InitialSOC_pct', 'FirstConnectedHour'
    ] + connect_cols_list

    # Include optional columns if they exist
    if 'MaxDischargeRateMW' in final_df.columns: required_sim_cols.append('MaxDischargeRateMW')
    if 'DischargeEfficiency' in final_df.columns: required_sim_cols.append('DischargeEfficiency')

    # Filter final DataFrame to include only required columns, handling missing ones gracefully
    final_cols_present = [col for col in required_sim_cols if col in final_df.columns]
    missing_cols = set(required_sim_cols) - set(final_cols_present)
    if missing_cols:
        print(f"Warning: Final DataFrame is missing expected simulation columns: {missing_cols}")

    final_df_processed = final_df[final_cols_present].copy()


    print(f"Final processed data shape (one row per EV): {final_df_processed.shape}")
    expected_evs = 1800
    actual_evs = final_df_processed.shape[0]
    if actual_evs < expected_evs * 0.9: # Allow for some filtering
         print(f"Warning: Processed data has only {actual_evs} EVs, significantly less than the expected {expected_evs}. Some EVs might have been filtered out due to missing/invalid data.")
    elif actual_evs > expected_evs:
         # This shouldn't happen if EV_ID generation is correct
         print(f"Warning: Processed data has {actual_evs} EVs, more than the expected {expected_evs}. Check EV_ID generation logic.")

    # Final check for NaNs in critical columns
    critical_cols_check = ['CapacityMWh', 'MaxChargeRateMW', 'ChargeEfficiency', 'Bus', 'InitialSOC_pct']
    nan_check = final_df_processed[[col for col in critical_cols_check if col in final_df_processed.columns]].isnull().sum()
    if nan_check.sum() > 0:
        print(f"Warning: NaNs detected in critical columns after processing:\n{nan_check[nan_check > 0]}")
        # Consider dropping these rows or implementing more robust filling if this occurs
        # final_df_processed.dropna(subset=[col for col in critical_cols_check if col in final_df_processed.columns], inplace=True)
        # print(f"Dropped additional rows with NaNs in critical columns. New shape: {final_df_processed.shape}")

    return final_df_processed


# --- Keep the run_detailed_simulation function as it was, with FutureWarning fix ---
def run_detailed_simulation(params, base_ev_data):
    """
    Runs a detailed EV charging simulation based on parameters.
    Expects base_ev_data in wide format (one row per EV).
    """
    if base_ev_data is None or base_ev_data.empty:
        print("Warning: Base EV data is empty for detailed simulation. Returning zero results.")
        return {
            'hourlyLoad': [0.0] * 24, 'peakLoad': 0.0, 'totalEnergy': 0.0,
            'concurrentEVs': [0] * 24, 'finalSOCs': [], 'avgFinalSoc': 0.0,
            'busLoads': {}, 'numEVsSimulated': 0
        }

    # --- 1. Filter EVs based on params ---
    sim_data = base_ev_data.copy()

    # Apply EV Percentage (random sampling if needed, or just take a fraction)
    n_total_evs = len(sim_data)
    n_selected_evs = int(round(n_total_evs * params['evPercentage'] / 100.0))
    if n_selected_evs == 0:
        print("Warning: EV Percentage resulted in 0 EVs selected.")
        return {'hourlyLoad': [0.0]*24, 'peakLoad': 0.0, 'totalEnergy': 0.0, 'concurrentEVs': [0]*24, 'finalSOCs': [], 'avgFinalSoc': 0.0, 'busLoads': {}, 'numEVsSimulated': 0}

    n_available = len(sim_data)
    n_to_sample = min(n_selected_evs, n_available)
    if n_to_sample < n_available:
        # Use sampling if fraction is less than 100%
        sim_data = sim_data.sample(n=n_to_sample, random_state=42) # Use random state for consistency if needed
    # else: use all data if 100%

    # Apply Bus Filter
    if params.get('selectedBuses') != 'all' and 'selectedBuses' in params:
        try:
            selected_bus_numbers = [int(b) for b in params['selectedBuses']]
            if 'Bus' in sim_data.columns:
                sim_data = sim_data[sim_data['Bus'].isin(selected_bus_numbers)]
            else:
                print("Warning: 'Bus' column not found in detailed sim data, cannot filter by bus.")
        except ValueError:
            print("Warning: Invalid bus selection format, using all available buses for detailed sim.")
        except Exception as e:
             print(f"Warning: Error applying bus filter: {e}. Using available buses.")


    # Apply EV Type Filter
    if 'selectedEvTypes' in params and params['selectedEvTypes']:
         try:
             selected_types = [int(t) for t in params['selectedEvTypes']]
             if 'VehicleType' in sim_data.columns:
                 # Check if filtering is necessary (don't filter if all types are selected)
                 available_types = sim_data['VehicleType'].unique()
                 if set(selected_types) != set(available_types):
                     sim_data = sim_data[sim_data['VehicleType'].isin(selected_types)]
             else:
                 print("Warning: 'VehicleType' column not found in detailed sim data, cannot filter by EV type.")
         except ValueError:
             print("Warning: Invalid EV type selection format, using all available types for detailed sim.")
         except Exception as e:
             print(f"Warning: Error applying EV Type filter: {e}. Using available types.")


    if sim_data.empty:
        print("Warning: No EVs remaining after filtering for detailed simulation. Returning zero results.")
        return {'hourlyLoad': [0.0]*24, 'peakLoad': 0.0, 'totalEnergy': 0.0, 'concurrentEVs': [0]*24, 'finalSOCs': [], 'avgFinalSoc': 0.0, 'busLoads': {}, 'numEVsSimulated': 0}

    # --- 2. Initialize Simulation State ---
    num_evs_simulated = len(sim_data)
    ev_states = sim_data.copy() # Work on a copy

    # Apply Initial SOC Adjustment
    if 'InitialSOC_pct' in ev_states.columns:
        ev_states['CurrentSOC_pct'] = (ev_states['InitialSOC_pct'] + params.get('socAdjustment', 0)).clip(0, 100)
    else:
        print("Warning: 'InitialSOC_pct' column missing for detailed sim. Assuming 0% initial SOC.")
        ev_states['CurrentSOC_pct'] = 0.0

    # Calculate Initial SOC in MWh
    if 'CapacityMWh' in ev_states.columns:
         # Ensure CapacityMWh is numeric and handle potential NaNs before calculation
         ev_states['CapacityMWh'] = pd.to_numeric(ev_states['CapacityMWh'], errors='coerce').fillna(0)
         ev_states['CurrentSOC_MWh'] = ev_states['CurrentSOC_pct'] / 100.0 * ev_states['CapacityMWh']
    else:
         print("Warning: 'CapacityMWh' column missing for detailed sim. Cannot calculate CurrentSOC_MWh.")
         ev_states['CurrentSOC_MWh'] = 0.0

    # Get parameters
    charge_logic = params.get('chargingLogic', 'immediate')
    charge_delay_hours = params.get('chargeDelay', 0)
    peak_start_hour = params.get('peakStart', 17) # Default peak start
    peak_end_hour = params.get('peakEnd', 20)     # Default peak end
    peak_reduction_pct = params.get('peakReduction', 50) # Default reduction
    soc_target_pct = params.get('socTarget', 80) # Default SOC target
    # Use the global limit from params, convert kW to MW
    global_charge_limit_kw = params.get('globalChargeLimit', 22) # Default 22 kW
    global_charge_limit_mw = global_charge_limit_kw / 1000.0
    schedule_shift = params.get('scheduleShift', 0)

    # Ensure ChargeEfficiency column exists and fill NaNs - Use the value from the data if present, else default
    if 'ChargeEfficiency' not in ev_states.columns:
        print("Warning: 'ChargeEfficiency' column missing in detailed sim data. Adding column with default 0.9.")
        ev_states['ChargeEfficiency'] = 0.9
    else:
        # Fill any remaining NaNs just in case
        ev_states['ChargeEfficiency'] = ev_states['ChargeEfficiency'].fillna(0.9)


    # Result arrays
    hourly_load_mw = np.zeros(24)
    concurrent_evs_count = np.zeros(24, dtype=int)
    # Get unique, non-NaN bus IDs present in the *filtered* simulation data
    bus_ids_in_sim = ev_states['Bus'].dropna().unique() if 'Bus' in ev_states.columns else []
    bus_hourly_loads = {int(bus_id): np.zeros(24) for bus_id in bus_ids_in_sim if pd.notna(bus_id)}


    # --- 3. Simulation Loop (24 hours) ---
    # Define required columns for the loop, check existence once
    required_loop_cols = [
        'CurrentSOC_MWh', 'CurrentSOC_pct', 'CapacityMWh', 'MaxChargeRateMW',
        'FirstConnectedHour', 'ChargeEfficiency', 'Bus'
    ] + [f'Connect_{h}' for h in range(24)]

    missing_loop_cols = [col for col in required_loop_cols if col not in ev_states.columns]
    if missing_loop_cols:
        print(f"FATAL Error in detailed sim: Missing essential columns for simulation loop: {missing_loop_cols}. Aborting simulation run.")
        # Return empty results to prevent crashes later
        return {'hourlyLoad': [0.0]*24, 'peakLoad': 0.0, 'totalEnergy': 0.0, 'concurrentEVs': [0]*24, 'finalSOCs': [], 'avgFinalSoc': 0.0, 'busLoads': {}, 'numEVsSimulated': num_evs_simulated} # Still report number attempted

    # Convert relevant columns to NumPy arrays for potentially faster access if DataFrame is large
    # Note: This might be overkill for 1800 rows, .at is often fast enough. Keeping .at for clarity.
    # soc_mwh_arr = ev_states['CurrentSOC_MWh'].to_numpy()
    # ... etc.

    for h in range(24):
        hour_load_mw = 0.0
        hour_concurrent_count = 0
        charging_indices_this_hour = [] # Track who charged

        # Iterate through EVs using index
        for idx in ev_states.index:
            # Access data using .at for speed and safety with mixed types
            ev_soc_mwh = ev_states.at[idx, 'CurrentSOC_MWh']
            ev_soc_pct = ev_states.at[idx, 'CurrentSOC_pct']
            ev_capacity_mwh = ev_states.at[idx, 'CapacityMWh']
            ev_max_charge_mw = ev_states.at[idx, 'MaxChargeRateMW']
            ev_first_connect = ev_states.at[idx, 'FirstConnectedHour'] # Can be -1 if never connects
            ev_efficiency = ev_states.at[idx, 'ChargeEfficiency']
            ev_bus = ev_states.at[idx, 'Bus'] # Could be NaN if data was missing

            charge_power_mw = 0.0 # Reset charge power for this EV for this hour

            # Check Connection Status based on shifted schedule
            connect_hour_shifted = (h - schedule_shift + 24) % 24
            connect_col = f'Connect_{connect_hour_shifted}'
            is_connected = ev_states.at[idx, connect_col] == 1

            # Check conditions: connected, not full, and capable of charging
            # Use a small tolerance for full check
            is_full = ev_soc_pct >= 99.9
            can_charge = ev_capacity_mwh > 1e-6 and ev_max_charge_mw > 1e-6 and ev_efficiency > 1e-6

            if is_connected and not is_full and can_charge:
                charge_flag = False # Should this EV attempt to charge based on logic?

                # Apply Charging Logic
                if charge_logic == 'immediate':
                    charge_flag = True
                elif charge_logic == 'delayed':
                    # Check if the current hour 'h' is at or after the delayed start time
                    # The EV must have connected at some point (ev_first_connect != -1)
                    if ev_first_connect != -1:
                        effective_connect_hour = (ev_first_connect + schedule_shift + 24) % 24 # Account for schedule shift in connection time
                        delayed_start_hour = (effective_connect_hour + charge_delay_hours) # No modulo here, delay can push past midnight
                        # Charge if current hour h is >= delayed start hour *within the 24h cycle*
                        # Need careful handling if delay pushes start past 24h mark conceptually
                        # Simplest: If first connection happened, charge if h >= first_connect + delay (modulo handled by connection check)
                        if h >= (ev_first_connect + charge_delay_hours): # Original simpler logic, relies on is_connected check
                             charge_flag = True

                elif charge_logic == 'offpeak':
                    # Default to charging, unless it's a peak hour where reduction applies
                    charge_flag = True
                elif charge_logic == 'soc_target':
                    # Charge only if current SOC is below the target
                    if ev_soc_pct < soc_target_pct:
                        charge_flag = True

                # If the logic allows charging, calculate the actual power
                if charge_flag:
                    # Determine the maximum rate possible this hour
                    # Limited by EV's max rate AND the global limit
                    max_rate_this_hour_mw = min(ev_max_charge_mw, global_charge_limit_mw)

                    # Apply Off-Peak Reduction if applicable
                    if charge_logic == 'offpeak':
                        is_peak_hour = False
                        # Handle wrap-around peak times (e.g., 22:00 - 02:00)
                        if peak_start_hour <= peak_end_hour:
                            is_peak_hour = peak_start_hour <= h <= peak_end_hour
                        else: # Peak wraps around midnight
                            is_peak_hour = h >= peak_start_hour or h <= peak_end_hour

                        if is_peak_hour:
                            max_rate_this_hour_mw *= (1.0 - peak_reduction_pct / 100.0)

                    # Clamp rate to non-negative
                    max_rate_this_hour_mw = max(0.0, max_rate_this_hour_mw)

                    # Calculate energy needed to reach full capacity
                    energy_needed_to_full_mwh = ev_capacity_mwh - ev_soc_mwh

                    # Calculate energy that *could* be added to the battery in 1 hour at the determined rate
                    max_energy_add_in_hour_mwh = max_rate_this_hour_mw * 1.0 * ev_efficiency

                    # Determine the energy we actually want to add (minimum of needed and possible)
                    desired_energy_add_mwh = max(0.0, min(energy_needed_to_full_mwh, max_energy_add_in_hour_mwh))

                    # Calculate the actual power draw required from the grid to add this energy
                    if desired_energy_add_mwh > 1e-9: # Avoid division by zero or tiny efficiency
                        actual_power_drawn_mw = desired_energy_add_mwh / ev_efficiency
                    else:
                        actual_power_drawn_mw = 0.0

                    # Ensure power drawn doesn't exceed the calculated max rate for the hour (redundant check, but safe)
                    actual_power_drawn_mw = min(actual_power_drawn_mw, max_rate_this_hour_mw)

                    # Update EV State (using .at for direct modification)
                    energy_added_to_battery_mwh = actual_power_drawn_mw * ev_efficiency
                    new_soc_mwh = min(ev_soc_mwh + energy_added_to_battery_mwh, ev_capacity_mwh) # Cap at capacity
                    # Update SOC percentage based on new MWh value
                    new_soc_pct = (new_soc_mwh / ev_capacity_mwh) * 100.0 if ev_capacity_mwh > 1e-6 else 0

                    ev_states.at[idx, 'CurrentSOC_MWh'] = new_soc_mwh
                    ev_states.at[idx, 'CurrentSOC_pct'] = new_soc_pct

                    charge_power_mw = actual_power_drawn_mw # This is the power drawn from grid

            # --- Update Aggregates for the hour ---
            if charge_power_mw > 1e-9: # Use a small threshold to count as charging
                hour_load_mw += charge_power_mw
                charging_indices_this_hour.append(idx) # Track who charged
                # Update bus load safely, checking if bus_id is valid and exists in our dict
                if pd.notna(ev_bus):
                    bus_id_int = int(ev_bus)
                    if bus_id_int in bus_hourly_loads:
                        bus_hourly_loads[bus_id_int][h] += charge_power_mw
                    # else: # Optional: Warn if EV's bus wasn't in the initial list (shouldn't happen with current setup)
                    #    print(f"Warning: EV {idx} belongs to Bus {bus_id_int}, which was not found in bus_hourly_loads dictionary.")


        # Store results for the hour
        hourly_load_mw[h] = hour_load_mw
        concurrent_evs_count[h] = len(charging_indices_this_hour) # Count how many EVs drew power > 0

    # --- 4. Calculate Final Metrics ---
    peak_load_mw = np.max(hourly_load_mw) if len(hourly_load_mw) > 0 else 0.0
    total_energy_mwh = np.sum(hourly_load_mw) if len(hourly_load_mw) > 0 else 0.0

    # Ensure CurrentSOC_pct exists before calculating final SOC stats
    if 'CurrentSOC_pct' in ev_states.columns:
        final_soc_list = ev_states['CurrentSOC_pct'].tolist()
        # Calculate average only if list is not empty
        avg_final_soc = np.mean(final_soc_list) if final_soc_list else 0.0
    else:
        final_soc_list = []
        avg_final_soc = 0.0 # Default if column was missing

    # Prepare final bus loads dictionary (convert numpy arrays to lists for JSON)
    bus_loads_final = {bus_id: loads.tolist() for bus_id, loads in bus_hourly_loads.items()}

    results = {
        'hourlyLoad': hourly_load_mw.tolist(),
        'peakLoad': peak_load_mw,
        'totalEnergy': total_energy_mwh,
        'concurrentEVs': concurrent_evs_count.tolist(),
        'finalSOCs': final_soc_list,
        'avgFinalSoc': avg_final_soc,
        'busLoads': bus_loads_final,
        'numEVsSimulated': num_evs_simulated # Include the number of EVs *after* filtering
    }
    return results


# --- Main Execution Block for Testing (Optional) ---
if __name__ == "__main__":
    print("--- Testing Simulation Engine ---")
    try:
        data_file_path = "EVs_cases_1800_EVs_for_33_Bus_MV_distribution_network.xlsx"
        print("\n1. Testing Data Loading and Preprocessing...")
        base_data = load_and_preprocess_data(filepath=data_file_path)

        # --- Add more detailed checks after loading ---
        print(f"\nBase data loaded and processed successfully. Shape: {base_data.shape}")
        if not base_data.empty:
            print("Sample processed data (first 3 rows):\n", base_data.head(3))
            print("\nProcessed data columns:\n", base_data.columns.tolist())
            # print("\nData types of processed columns:\n", base_data.dtypes) # Can be verbose
            print("\nChecking for NaNs in key simulation columns:")
            key_cols_check = [
                'EV_ID', 'VehicleType', 'ChargeEfficiency', 'CapacityMWh',
                'MaxChargeRateMW', 'Bus', 'InitialSOC_pct', 'MinChargeLimitMWh',
                'FirstConnectedHour'
            ] + [f'Connect_{h}' for h in range(24)]
            key_cols_present = [col for col in key_cols_check if col in base_data.columns]
            nan_counts = base_data[key_cols_present].isnull().sum()
            print(nan_counts[nan_counts > 0]) # Only print columns with NaNs

            # --- Run Simulation Test ---
            print("\n2. Testing Detailed Simulation Run...")
            test_params = {
                'evPercentage': 50, # Test with a subset
                'selectedBuses': [1, 5, 10], # Test bus filtering
                'selectedEvTypes': [1, 3], # Test type filtering
                'chargingLogic': 'offpeak', # Test a specific logic
                'chargeDelay': 2, # Relevant for 'delayed'
                'peakStart': 18, # Relevant for 'offpeak'
                'peakEnd': 21,   # Relevant for 'offpeak'
                'peakReduction': 75, # Relevant for 'offpeak'
                'socTarget': 90, # Relevant for 'soc_target'
                'globalChargeLimit': 11, # kW - Test a lower limit
                'scheduleShift': 2, # Test schedule shift
                'socAdjustment': -10, # Test SOC adjustment
                # 'chargingEfficiency': 90, # This is now read from data, param ignored by sim
            }

            print("\nRunning simulation with test parameters:")
            print(test_params)
            sim_results = run_detailed_simulation(test_params, base_data)

            print("\nSimulation Results:")
            print(f"  Number of EVs Simulated: {sim_results.get('numEVsSimulated', 'N/A')}")
            print(f"  Peak Load (Sim): {sim_results.get('peakLoad', 0.0):.3f} MW")
            print(f"  Total Energy (Sim): {sim_results.get('totalEnergy', 0.0):.3f} MWh")
            avg_soc = sim_results.get('avgFinalSoc', float('nan'))
            print(f"  Average Final SOC (Sim): {avg_soc:.1f}%" if pd.notna(avg_soc) else "N/A")
            print(f"  Max Concurrent EVs (Sim): {max(sim_results.get('concurrentEVs', [0]))}")
            print(f"  Number of Buses Tracked: {len(sim_results.get('busLoads', {}))}")
            # print("  Hourly Load (Sim, MW):", [f"{x:.3f}" for x in sim_results.get('hourlyLoad', [])]) # Optional: Full load profile
            print("  Hourly Concurrent EVs (Sim):", sim_results.get('concurrentEVs', []))

        else:
            print("Processed base_data is empty. Cannot run simulation test.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"Please ensure '{data_file_path}' is in the correct directory.")
    except ValueError as e:
         print(f"\nData Processing Error: {e}")
         traceback.print_exc()
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing: {e}")
        traceback.print_exc() # Print full traceback for debugging
