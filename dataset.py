import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loader_test import load_all_data
from functions import bandpass_filter, notch_filter, segment_epochs, extract_frequency_bands, compute_power_features, \
    compute_theta_beta_ratio, compute_fooof_features, compute_dfa, is_seizure

EPOCH_DURATION = 58
# Load all EEG data using loader_test.py
data_list, annotation_list = load_all_data(['eeg'], tsv_file="net/datasets/SZ2_training.tsv")

# Initialize list to store summary statistics
all_features = []

# Iterate through each recording
for rec_idx, data in enumerate(data_list):
    print(f"Processing recording {rec_idx+1}/{len(data_list)}")

    eeg_data = data.data  # This is a list of NumPy arrays (one per channel)
    sampling_rate = data.fs  # Sampling frequency
    channel_names = data.channels  # Channel names
    seizure_events = annotation_list[rec_idx].events
    j = 0
    # Iterate over each channel
    for ch_idx, ch_data in enumerate(eeg_data):
        ch_name = channel_names[ch_idx]  # Get the channel name
        duration = len(ch_data) / sampling_rate[j]  # Duration in seconds
        sfreq = sampling_rate[j]
        j = j + 1
        
        # Preprocessing: Bandpass filter, Notch filter
        bandpassed = bandpass_filter(ch_data, sfreq)
        notch_filtered = notch_filter(bandpassed, sfreq)
        
        # Segment the data into 1-second epochs
        segmented = segment_epochs(notch_filtered, sfreq, EPOCH_DURATION)
        
        # Compute frequency band features
        bands = extract_frequency_bands(segmented, sfreq)
        
        # Compute power features
        power_features = compute_power_features(segmented, sfreq)
        
        # Compute Theta/Beta Ratio
        theta_beta_ratios = compute_theta_beta_ratio(segmented, sfreq)
        
        # Compute peak alpha and 1/f exponent using FOOOF
        alpha_peaks, one_over_f_exponents = compute_fooof_features(segmented, sfreq)
        
        # Compute DFA exponent
        dfa_exponents = compute_dfa(segmented)

        # Combine all features into a dictionary for each segment
        for i in range(len(segmented)):
            epoch_start = i * EPOCH_DURATION  # Compute actual start time of epoch
            epoch_end = (i + 1) * EPOCH_DURATION
            feature_dict = {
                "recording": rec_idx + 1,
                "channel": ch_name,
                "epoch": i + 1,
                "theta_beta_ratio": theta_beta_ratios[i],
                "alpha_peak": alpha_peaks[i],
                "one_over_f": one_over_f_exponents[i],
                "dfa_exponent": dfa_exponents[i],
                "seizure": is_seizure(epoch_start, epoch_end, seizure_events)
            }
            
            # Add power features (band powers) to the dictionary
            feature_dict.update(power_features[i])  # Add band power features

            # If seizure labels exist, add them here (You need a function to get labels)
            #feature_dict["seizure"] = get_seizure_label(rec_idx, i)

            all_features.append(feature_dict)

# Convert list of dictionaries to a DataFrame
df_features = pd.DataFrame(all_features)

# Save the DataFrame to a CSV file
df_features.to_csv("eeg_features.csv", index=False)

print("Feature extraction completed! Features saved to eeg_features.csv")
