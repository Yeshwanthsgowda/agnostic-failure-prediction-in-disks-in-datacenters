import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(csv_path='dataset/ST12000NM0007_last_10_day_all_q_raw.csv', output_dir='research'):
    print(f"Loading REAL data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Sort strictly by serial_number and date
    print("Sorting data chronologically by serial number...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['serial_number', 'date']).reset_index(drop=True)

    # 2. Filter to ONLY keep serial_numbers that have EXACTLY 10 rows
    counts = df.groupby('serial_number').size()
    valid_serials = counts[counts == 10].index
    
    original_len = len(df)
    df = df[df['serial_number'].isin(valid_serials)].reset_index(drop=True)
    print(f"Filtered out {original_len - len(df)} rows that did not have exactly 10 days of history.")

    # Define columns
    meta_cols = ['date', 'serial_number', 'model', 'capacity_bytes']
    target_col = 'failure'
    
    # Only keep the raw smart stats for features
    feature_cols = [c for c in df.columns if c.startswith('smart_') and c.endswith('_raw')]
    
    # Drop zero-variance columns (features that are entirely zero across all drives)
    non_constant = [c for c in feature_cols if df[c].nunique() > 1]
    feature_cols = non_constant
    print(f"Using {len(feature_cols)} non-constant SMART features.")

    num_drives = len(df) // 10
    num_features = len(feature_cols)

    # 3. SCENARIO 1 (LSTM): 3D Array
    print("Reshaping for LSTM (3D Array)...")
    X_3d = df[feature_cols].values.reshape((num_drives, 10, num_features))
    
    y_3d = df.groupby('serial_number')[target_col].max().values
    serial_numbers = df['serial_number'].unique()

    # 4. SCENARIO 2 (XGBoost): Flattened 2D Array
    print("Flattening for XGBoost (2D Array)...")
    df['day_index'] = df.groupby('serial_number').cumcount() + 1
    df_pivoted = df.pivot(index='serial_number', columns='day_index', values=feature_cols)
    
    df_pivoted.columns = [f"{feat}_day_{day}" for feat, day in df_pivoted.columns]
    df_pivoted = df_pivoted.reindex(serial_numbers)
    X_flat = df_pivoted.values
    y_flat = y_3d

    # 5. Calculate 'scale_pos_weight'
    n_pos = np.sum(y_flat == 1)
    n_neg = np.sum(y_flat == 0)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Class Distribution -> Failing: {n_pos}, Healthy: {n_neg}")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # 6. Train/Test Split
    print("Splitting data at the serial_number level (80/20)...")
    indices = np.arange(num_drives)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_flat)

    data_artifacts = {
        'X_train_3d': X_3d[train_idx], 'X_test_3d': X_3d[test_idx],
        'y_train': y_flat[train_idx],  'y_test': y_flat[test_idx],
        'X_train_flat': X_flat[train_idx], 'X_test_flat': X_flat[test_idx],
        'scale_pos_weight': scale_pos_weight,
        'feature_names_flat': df_pivoted.columns.tolist()
    }

    # 7. Save
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, 'prepared_data.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(data_artifacts, f)
    
    print(f"Pipeline complete. Artifacts saved to {out_file}")

if __name__ == "__main__":
    prepare_data()
