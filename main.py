from src.data_preprocessing import load_data, preprocess_data
from src.model_training import init_model, train_model, save_model

def run_pipeline(data_path, model_path):
    """
    Executes the full machine learning pipeline.
    """
    print("--- Sentinel-1 Pipeline Started ---")
    
    # 1. Ingestion & Preprocessing
    data = load_data(data_path)
    if data is None:
        return
        
    X, y = preprocess_data(data)
    
    # 2. Training
    model = init_model()
    model = train_model(model, X, y)
    
    # 3. Serialization
    save_model(model, model_path)
    
    print("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    run_pipeline('transaction_dataset.csv', 'eth_fraud_model.pkl')
