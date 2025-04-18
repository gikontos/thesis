import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from loader_test import load_all_data
from functions import (
    bandpass_filter, notch_filter, segment_epochs, extract_statistical_features,
    extract_temporal_features, extract_complexity_features,
    compute_power_features, compute_theta_beta_ratio,
    generate_ft_surrogate, is_seizure
)

EPOCH_DURATION = 58  # seconds
SECOND_DURATION = 1  # 1-second segments
PARTIAL_FEATURES_FILE = "progress/features_partial_2.csv"
PROGRESS_LOG = "progress/progress_log_2.txt"

# Ensure directories exist
os.makedirs("progress", exist_ok=True)
os.makedirs("datasets", exist_ok=True)

# Load completed recordings if exists
if os.path.exists(PROGRESS_LOG):
    with open(PROGRESS_LOG) as f:
        completed_recordings = set(int(line.strip()) for line in f)
else:
    completed_recordings = set()


def process_single_recording(args):
    rec_idx, data, annotation, completed = args
    rec_number = rec_idx + 1

    if rec_number in completed:
        return None  # Skip if already processed

    all_segments = []
    eeg_data = data.data
    sampling_rate = data.fs
    channel_names = data.channels
    seizure_events = annotation.events

    for ch_idx, ch_data in enumerate(eeg_data):
        ch_name = channel_names[ch_idx]
        sfreq = sampling_rate[ch_idx]

        bandpassed = bandpass_filter(ch_data, sfreq)
        filtered = notch_filter(bandpassed, sfreq)
        epochs = segment_epochs(filtered, sfreq, EPOCH_DURATION)

        for i, epoch in enumerate(epochs):
            for signal_variant, aug_type in [(epoch, "original"), (generate_ft_surrogate(epoch), "ft_surrogate")]:
                one_sec_segments = segment_epochs(signal_variant, sfreq, SECOND_DURATION)
                power_features = compute_power_features(one_sec_segments, sfreq)
                theta_beta_ratios = compute_theta_beta_ratio(one_sec_segments, sfreq)

                for sec_idx, segment in enumerate(one_sec_segments):
                    start_time = i * EPOCH_DURATION + sec_idx
                    end_time = start_time + 1
                    label = is_seizure(start_time, end_time, seizure_events)

                    f_stat = extract_statistical_features(segment)
                    f_temp = extract_temporal_features(segment)
                    f_comp = extract_complexity_features(segment)

                    power = power_features[sec_idx]
                    theta_beta = theta_beta_ratios[sec_idx]

                    feature = {
                        "recording": rec_number,
                        "channel": ch_name,
                        "epoch": i + 1,
                        "second": sec_idx + 1,
                        "augmentation": aug_type,
                        "seizure": label,
                        "theta_beta_ratio": theta_beta,
                        **f_stat, **f_temp, **f_comp, **power
                    }

                    all_segments.append(feature)

    return rec_number, all_segments


# Load EEG data
print("Loading EEG data...")
data_list, annotation_list = load_all_data(['eeg'], tsv_file="helpers/net/datasets/SZ2_training_toy.tsv")

# Set number of workers
max_workers = min(8, cpu_count())  # Adjust this based on instance size

# Run processing in parallel
print(f"Processing {len(data_list)} recordings using {max_workers} parallel workers...")

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    tasks = [
        executor.submit(process_single_recording, (rec_idx, data, annotation_list[rec_idx], completed_recordings))
        for rec_idx, data in enumerate(data_list)
    ]

    for future in as_completed(tasks):
        result = future.result()
        if result is None:
            continue
        rec_number, all_segments = result

        # Save partial results
        df_part = pd.DataFrame(all_segments)
        write_header = not os.path.exists(PARTIAL_FEATURES_FILE)
        df_part.to_csv(PARTIAL_FEATURES_FILE, mode='a', header=write_header, index=False)

        # Update progress log
        with open(PROGRESS_LOG, 'a') as f:
            f.write(f"{rec_number}\n")

        print(f"Saved features for recording {rec_number}.")


# Create final datasets
print("Combining all partial features...")

df = pd.read_csv(PARTIAL_FEATURES_FILE)
seizure_df = df[df["seizure"] == 1]
non_seizure_df = df[df["seizure"] == 0]

ratios = [2, 10, 100]
datasets = {}

for ratio in ratios:
    num_seizure = len(seizure_df)
    num_non_seizure = min(len(non_seizure_df), ratio * num_seizure)
    sampled_non_seizure = non_seizure_df.sample(n=num_non_seizure, random_state=42)
    combined = pd.concat([seizure_df, sampled_non_seizure], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42)

    datasets[f"features_5t15_aug_1_{ratio}.csv"] = combined
    datasets[f"features_5t15_noaug_1_{ratio}.csv"] = combined[combined['augmentation'] == 'original']

for name, df_out in datasets.items():
    df_out.to_csv(f"datasets/{name}", index=False)

print(" All datasets saved successfully.")
