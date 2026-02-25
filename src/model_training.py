from sklearn.ensemble import RandomForestClassifier
import joblib

def init_model(max_depth=5, random_state=42):
    """
    Initializes the Random Forest Classifier with specified parameters.
    """
    model = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    return model

def train_model(model, X, y):
    """
    Trains the model on the feature matrix X and target vector y.
    """
    print("Training Random Forest Classifier...")
    model.fit(X, y)
    return model

def save_model(model, file_path):
    """
    Serializes the model artifact to a disk file.
    """
    try:
        joblib.dump(model, file_path)
        print(f"Model artifact saved to: {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
