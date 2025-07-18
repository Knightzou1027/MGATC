import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

# --- Configuration ---
data_path = r"data/station_od_idv_1018_attributes.xlsx" # Adjust path if needed

# --- Helper Function ---
def calculate_overlap_proportion(interval_start_dt, interval_end_dt, block_start_dt, block_end_dt):
    """Calculates the proportion of the interval that overlaps with the block."""
    total_duration_seconds = (interval_end_dt - interval_start_dt).total_seconds()
    if total_duration_seconds <= 0: return 0.0
    overlap_start = max(interval_start_dt, block_start_dt)
    overlap_end = min(interval_end_dt, block_end_dt)
    overlap_duration_seconds = (overlap_end - overlap_start).total_seconds()
    if overlap_duration_seconds <= 0: return 0.0
    else: return min(overlap_duration_seconds / total_duration_seconds, 1.0)

# --- Core Processing ---
try:
    # 1. Load Data
    df = pd.read_excel(data_path)

    # 2. Handle missing 'age'
    if df['age'].dtype == 'object':
         df['age'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df.dropna(subset=['age'], inplace=True)

    # 3. Parse Time Columns
    df['s_time_dt'] = pd.to_datetime(df['s_time'], errors='coerce')
    df['e_time_dt'] = pd.to_datetime(df['e_time'], errors='coerce')

    # 4. Calculate Activity End Time (a_time_dt) using staytime in hours
    df['staytime_hours'] = pd.to_numeric(df['staytime'], errors='coerce').fillna(0)
    df['a_time_dt'] = df['e_time_dt'] + pd.to_timedelta(df['staytime_hours'], unit='h', errors='coerce')

    # 5. Create 'a_time_hhmm' column
    df['a_time_hhmm'] = df['a_time_dt'].dt.strftime('%H:%M').fillna('NaT')

    # 6. Drop rows with NaT datetimes
    df.dropna(subset=['s_time_dt', 'e_time_dt', 'a_time_dt'], inplace=True)
    # Update HH:MM in case NaT rows were dropped after initial creation
    df['a_time_hhmm'] = df['a_time_dt'].dt.strftime('%H:%M')


    # 7. Calculate Time Block Features
    if not df.empty:
        num_blocks = 48
        se_feature_cols = [f'se_block_{i}' for i in range(num_blocks)]
        ea_feature_cols = [f'ea_block_{i}' for i in range(num_blocks)]

        se_features_array = np.zeros((len(df), num_blocks))
        ea_features_array = np.zeros((len(df), num_blocks))

        for i, row in enumerate(df.itertuples(index=False)):
            s_dt = row.s_time_dt
            e_dt = row.e_time_dt
            a_dt = row.a_time_dt
            ref_date = s_dt.date()

            for j in range(num_blocks):
                block_start_minute = j * 30
                block_end_minute = (j + 1) * 30
                start_time_of_day = time(hour=block_start_minute // 60, minute=block_start_minute % 60)
                block_start_dt = datetime.combine(ref_date, start_time_of_day)

                if block_end_minute == 24 * 60:
                     block_end_dt = datetime.combine(ref_date + timedelta(days=1), time(0, 0))
                else:
                     end_time_of_day = time(hour=block_end_minute // 60, minute=block_end_minute % 60)
                     block_end_dt = datetime.combine(ref_date, end_time_of_day)

                se_features_array[i, j] = calculate_overlap_proportion(s_dt, e_dt, block_start_dt, block_end_dt)
                ea_features_array[i, j] = calculate_overlap_proportion(e_dt, a_dt, block_start_dt, block_end_dt)

        # Assign features back
        df = df.assign(**{col: se_features_array[:, i] for i, col in enumerate(se_feature_cols)})
        df = df.assign(**{col: ea_features_array[:, i] for i, col in enumerate(ea_feature_cols)})

    # 8. Final Cleanup
    df = df.drop(columns=['s_time_dt', 'e_time_dt', 'a_time_dt', 'staytime_hours'])
    print(df.head())

    # --- Output (Optional) ---
    print(f"Processing complete. Final DataFrame shape: {df.shape}")
    # print("\nFinal DataFrame head:")
    # display_cols = ['order', 's_time', 'e_time', 'staytime', 'age', 'a_time_hhmm'] + [f'se_block_{i}' for i in range(2)] + [f'ea_block_{i}' for i in range(2)]
    # print(df[display_cols].head())

    # --- Save Result (Optional) ---
    output_path = 'od_data_timeblock.csv' # Adjust path if needed
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")

except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}")
except Exception as e:
    print(f"An unexpected error occurred during processing: {e}")
    # import traceback
    # traceback.print_exc() # Uncomment for detailed error traceback