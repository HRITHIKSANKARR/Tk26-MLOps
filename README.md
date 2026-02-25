# Sentinel-1: Ethereum Fraud Detection Pipeline

Modular machine learning pipeline for predicting Ethereum wallet fraud based on transaction behavior.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training Pipeline**:
   This will clean the data, train a Random Forest model, and save it as `eth_fraud_model.pkl`.
   ```bash
   python3 main.py
   ```

3. **Run Inference App**:
   Launch the Streamlit dashboard to test wallet addresses.
   ```bash
   streamlit run app.py
   ```

## Directory Structure
- `src/`: Core logic modules.
  - `data_preprocessing.py`: Feature selection and cleaning.
  - `model_training.py`: Model definition and training logic.
- `main.py`: Orchestration script.
- `app.py`: Streamlit web interface.
- `Sentinel1_Kaggle_Pipeline.ipynb`: All-in-one notebook for Kaggle.
- `transaction_dataset.csv`: Source dataset.
- `eth_fraud_model.pkl`: Generated model artifact.
