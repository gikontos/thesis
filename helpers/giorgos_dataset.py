import pandas as pd
import numpy as np
import random
from loader_test import load_all_data
from functions import (
    bandpass_filter, notch_filter, segment_epochs, extract_statistical_features,
    extract_temporal_features, extract_complexity_features,
    compute_power_features, compute_theta_beta_ratio,
    generate_ft_surrogate, is_seizure
)

EPOCH_DURATION = 58  # seconds
SECOND_DURATION = 1   # 1-second segments

# Load EEG data
data_list, annotation_list = load_all_data(['eeg'], tsv_file="helpers/net/datasets/SZ2_training_toy.tsv")

# List to store all segments and features
all_segments = []

for rec_idx, data in enumerate(data_list):
    print(f"Processing recording {rec_idx + 1}/{len(data_list)}")

    eeg_data = data.data
    sampling_rate = data.fs
    channel_names = data.channels
    seizure_events = annotation_list[rec_idx].events

    for ch_idx, ch_data in enumerate(eeg_data):
        ch_name = channel_names[ch_idx]
        sfreq = sampling_rate[ch_idx]

        # Filter the signal
        bandpassed = bandpass_filter(ch_data, sfreq)
        filtered = notch_filter(bandpassed, sfreq)

        # Segment into full-length epochs
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
                        "recording": rec_idx + 1,
                        "channel": ch_name,
                        "epoch": i + 1,
                        "second": sec_idx + 1,
                        "augmentation": aug_type,
                        "seizure": label,
                        "theta_beta_ratio": theta_beta,
                        **f_stat, **f_temp, **f_comp, **power
                    }

                    all_segments.append(feature)

# Convert to DataFrame
df = pd.DataFrame(all_segments)

# Create datasets based on seizure:non-seizure ratios
seizure_df = df[df["seizure"] == 1]
non_seizure_df = df[df["seizure"] == 0]

ratios = [2, 10, 100]
datasets = {}

for ratio in ratios:
    num_seizure = len(seizure_df)
    num_non_seizure = min(len(non_seizure_df), ratio * num_seizure)
    sampled_non_seizure = non_seizure_df.sample(n=num_non_seizure, random_state=42)
    combined = pd.concat([seizure_df, sampled_non_seizure], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42)  # shuffle

    # With augmentation
    with_aug = combined
    datasets[f"features_aug_1_{ratio}.csv"] = with_aug

    # Without augmentation (remove ft_surrogate)
    without_aug = combined[combined['augmentation'] == 'original']
    datasets[f"features_noaug_1_{ratio}.csv"] = without_aug

# Save all datasets
for name, df_out in datasets.items():
    df_out.to_csv(f"datasets/{name}", index=False)

print("All datasets saved with specified seizure:non-seizure ratios and augmentation variants.")
