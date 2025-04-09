import os
import joblib
import numpy as np
from generator import SequentialGenerator
from utils import get_events

def predict(submission_path, recording):
    model_path = os.path.join(submission_path, 'seizft_xgb_model.pkl')
    model = joblib.load(model_path)

    generator = SequentialGenerator(recording)
    X_input = np.array([generator[i] for i in range(len(generator))])
    
    # Probabilities for class 1 (seizure)
    y_pred = model.predict_proba(X_input)[:, 1]
    
    # Convert to seizure events
    events = get_events(y_pred, recording)
    
    return events
