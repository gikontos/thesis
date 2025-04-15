from model import predict
from helpers.loader_test import load_all_data
import os
import csv

# Output folders
os.makedirs("submit_output", exist_ok=True)
os.makedirs("reference", exist_ok=True)

# Load validation data
data_list, annotations_list = load_all_data(['eeg'], tsv_file="helpers/net/datasets/SZ2_validation_test.tsv")

for i, recording in enumerate(data_list):
    print(f"Predicting on recording {i+1}/{len(data_list)}")
    predicted_events = predict("", recording)

    # Save predicted events to CSV
    recording_id = f"rec_{i+1:03d}_baseline_test2.csv"
    pred_path = os.path.join("submit_output", recording_id)
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        for event in predicted_events:
            writer.writerow([float(f"{event[0]:.2f}"), float(f"{event[1]:.2f}")])

    # Save true events to CSV with duration at the end
    true_events = annotations_list[i].events
    rec_duration = len(recording.data[0]) / recording.fs[0]  # assumes all channels same length/fs
    ref_path = os.path.join("reference", recording_id)
    with open(ref_path, "w", newline="") as f:
        writer = csv.writer(f)
        for event in true_events:
            writer.writerow([float(f"{event[0]:.2f}"), float(f"{event[1]:.2f}")])
        writer.writerow([0.0, rec_duration])  # final row = [0, total_duration]
