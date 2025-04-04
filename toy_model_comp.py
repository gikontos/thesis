import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

# Load dataset
df = pd.read_csv("eeg_features_toy.csv")

# Drop non-numeric columns
df = df.drop(columns=["recording", "epoch", "channel", "second","alpha_peak"])

# Handle missing values (if any)
#df = df.dropna()

# Separate features & labels
X = df.drop(columns=["seizure"])
y = df["seizure"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split (use fixed random_state and save indices)
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X_scaled, y, df.index, test_size=0.2, random_state=42, stratify=y
)

# Save test indices to ensure consistent validation set
np.savetxt("validation_indices.txt", test_idx, fmt="%d")

print("Validation indices saved! Use this file to create the reference set.")

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on validation set
y_pred_prob = clf.predict_proba(X_test)[:, 1]  # Get probability of seizure class

# Apply threshold to get binary predictions
threshold = 0.5  # Adjust if needed
y_pred = (y_pred_prob >= threshold).astype(int)

# Convert predictions into event format
def convert_to_events(y_pred, fs=1):
    events = []
    in_event = False
    start_time = 0

    for i, val in enumerate(y_pred):
        if val == 1 and not in_event:
            start_time = i / fs
            in_event = True
        elif val == 0 and in_event:
            end_time = i / fs
            events.append([start_time, end_time])
            in_event = False

    # Handle case where the last event reaches the end
    if in_event:
        events.append([start_time, len(y_pred) / fs])

    return events

# Convert test predictions to event format
predicted_events = convert_to_events(y_pred)

# Save results in competition format
output_dir = "submit_output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "predictions.csv")

with open(output_file, "w") as f:
    for event in predicted_events:
        f.write(f"{event[0]},{event[1]}\n")

print(f"Predictions saved to {output_file}")
