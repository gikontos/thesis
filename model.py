import os
import joblib
import numpy as np
from giorgos_generator import SequentialGenerator
from utils import get_events

def predict(submission_path, recording):
    model_path = os.path.join(submission_path, 'xgb_model_aug_1_2.pkl')
    model = joblib.load(model_path)

    generator = SequentialGenerator(recording)
    X_input = np.array([generator[i] for i in range(len(generator))])
    
    # Probabilities for class 1 (seizure)
    y_pred = model.predict_proba(X_input)[:, 1]
    
    # Convert to seizure events
    events = get_events(y_pred, recording)
    
    return events

# def predict(submission_path, recording):
#     # Load model and scaler
#     model_path = os.path.join(submission_path, 'rf_model.pkl')
#     scaler_path = os.path.join(submission_path, 'scaler.pkl')

#     model = joblib.load(model_path)
#     scaler = joblib.load(scaler_path)

#     # Generate features
#     generator = SequentialGenerator(recording)
#     X_input = np.array([generator[i] for i in range(len(generator))])

#     # Apply scaling
#     X_scaled = scaler.transform(X_input)

#     # Get class probabilities
#     y_pred = model.predict_proba(X_scaled)[:, 1]

#     # Convert to seizure events
#     events = get_events(y_pred, recording)
    
#     return events
