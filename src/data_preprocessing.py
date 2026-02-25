import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Loads the transaction dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Selects target features and cleans the data as per PRD requirements.
    Behavioral Pillars:
    - Velocity: Avg min between sent tnx
    - Lifespan: Time Diff between first and last (Mins)
    - Outflow: Sent tnx
    - Inflow: Received Tnx
    """
    # Features requested in PRD
    selected_features = [
        'Avg min between sent tnx',
        'Time Diff between first and last (Mins)',
        'Sent tnx',
        'Received Tnx'
    ]
    target = 'FLAG'
    
    # Audit for NaNs and drop them
    df_clean = df[selected_features + [target]].copy()
    initial_len = len(df_clean)
    df_clean = df_clean.dropna()
    final_len = len(df_clean)
    
    if initial_len > final_len:
        print(f"Dropped {initial_len - final_len} rows containing NaN values.")
        
    X = df_clean[selected_features]
    y = df_clean[target]
    
    return X, y

if __name__ == "__main__":
    # Quick test
    data = load_data('transaction_dataset.csv')
    if data is not None:
        X, y = preprocess_data(data)
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
