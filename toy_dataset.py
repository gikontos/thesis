import pandas as pd
import numpy as np
import random
from loader_test import load_all_data
from functions import (
    bandpass_filter, notch_filter, segment_epochs, extract_frequency_bands,
    compute_power_features, compute_theta_beta_ratio, compute_fooof_features,
    compute_dfa, is_seizure
)

EPOCH_DURATION = 58  # Full epoch duration (seconds)
SECOND_DURATION = 1   # 1-second segments

# Load EEG data
data_list, annotation_list = load_all_data(['eeg'], tsv_file="net/datasets/SZ2_training_toy.tsv")

# List to store features
all_features = []

# Iterate through recordings
for rec_idx, data in enumerate(data_list):
    print(f"Processing recording {rec_idx+1}/{len(data_list)}")

    eeg_data = data.data  # List of NumPy arrays (one per channel)
    sampling_rate = data.fs  # Sampling frequency
    channel_names = data.channels  # Channel names
    seizure_events = annotation_list[rec_idx].events
    
    # Process selected epochs
    for ch_idx, ch_data in enumerate(eeg_data):
        ch_name = channel_names[ch_idx]  # Get channel name
        sfreq = sampling_rate[ch_idx]
        
        # Preprocessing: Bandpass and Notch filter
        bandpassed = bandpass_filter(ch_data, sfreq)
        notch_filtered = notch_filter(bandpassed, sfreq)
        
        # Segment the data into full-length epochs
        segmented = segment_epochs(notch_filtered, sfreq, EPOCH_DURATION)

        # Identify seizure and non-seizure epochs
        seizure_epochs = []
        non_seizure_epochs = []
        for j in range(len(segmented)):
            epoch_start = j * EPOCH_DURATION
            epoch_end = (j + 1) * EPOCH_DURATION
            if is_seizure(epoch_start, epoch_end, seizure_events):
                seizure_epochs.append(j)
            else:
                non_seizure_epochs.append(j)

        # Randomly select 66 non-seizure epochs
        random_non_seizure_epochs = random.sample(non_seizure_epochs, min(66, len(non_seizure_epochs)))
        selected_epochs = set(seizure_epochs + random_non_seizure_epochs)
        
        for i in selected_epochs:
            full_epoch = segmented[i]
            
            # Split full epoch into 1s segments (efficiently in one step)
            one_sec_segments = segment_epochs(full_epoch, sfreq, SECOND_DURATION)
            
            # Compute all features at once for the full epoch (batch processing)
            power_features = compute_power_features(one_sec_segments, sfreq)
            theta_beta_ratios = compute_theta_beta_ratio(one_sec_segments, sfreq)
            alpha_peaks, one_over_f_exponents = compute_fooof_features(one_sec_segments, sfreq)
            dfa_exponents = compute_dfa(one_sec_segments)
            
            # Iterate through all 1s segments and save the features
            for sec, (power, theta_beta, alpha_peak, one_over_f, dfa_exp) in enumerate(zip(
                power_features, theta_beta_ratios, alpha_peaks, one_over_f_exponents, dfa_exponents
            )):
                absolute_sec_start = (i * EPOCH_DURATION) + sec
                absolute_sec_end = absolute_sec_start + 1

                feature_dict = {
                    "recording": rec_idx + 1,
                    "channel": ch_name,
                    "epoch": i + 1,
                    "second": sec + 1,
                    "theta_beta_ratio": theta_beta,
                    "alpha_peak": alpha_peak,
                    "one_over_f": one_over_f,
                    "dfa_exponent": dfa_exp,
                    "seizure": is_seizure(absolute_sec_start, absolute_sec_end, seizure_events),
                }

                feature_dict.update(power)  # Add power features
                all_features.append(feature_dict)

# Convert list to DataFrame
df_features = pd.DataFrame(all_features)

# Save to CSV
df_features.to_csv("eeg_features_toy.csv", index=False)
print("Feature extraction completed! Features saved to eeg_features_toy.csv")
