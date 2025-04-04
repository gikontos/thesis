import pandas as pd
import numpy as np
import os

# Load full dataset
df = pd.read_csv("eeg_features_toy.csv")

# Load validation indices from saved file
val_indices = np.loadtxt("validation_indices.txt", dtype=int)

# Extract only validation (test) set events
df_val = df.loc[val_indices, ["seizure"]]

# Convert to event format (if needed)
def convert_to_events(y_series, fs=1):
    events = []
    in_event = False
    start_time = 0

    for i, val in enumerate(y_series):
        if val == 1 and not in_event:
            start_time = i / fs
            in_event = True
        elif val == 0 and in_event:
            end_time = i / fs
            events.append([start_time, end_time])
            in_event = False

    if in_event:
        events.append([start_time, len(y_series) / fs])

    return events

# Convert validation labels into event list
true_events = convert_to_events(df_val["seizure"].values)

# Calculate recording duration
recording_duration = len(df_val)  # fs = 1 Hz, so 1 index = 1 second

# Save to CSV (reference file)
output_dir = "reference"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "validation_reference.csv"), "w") as f:
    for event in true_events:
        f.write(f"{event[0]},{event[1]}\n")
    f.write(f"0,{recording_duration}\n")  # <- Add final row as required by score.py

print("Reference file saved to 'reference/validation_reference.csv'.")
