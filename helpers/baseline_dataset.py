import pandas as pd
import numpy as np
import random
from loader_test import load_all_data
from functions import (
    bandpass_filter, notch_filter, segment_epochs, is_seizure, extract_complexity_features, extract_spectral_features, \
    extract_statistical_features, extract_temporal_features, generate_ft_surrogate
)

EPOCH_DURATION = 58  # seconds
SECOND_DURATION = 1   # 1-second segments

# Load EEG data
data_list, annotation_list = load_all_data(['eeg'], tsv_file="net/datasets/SZ2_training_toy.tsv")

all_features = []

for rec_idx, data in enumerate(data_list):
    print(f"Processing recording {rec_idx + 1}/{len(data_list)}")
    eeg_data = data.data
    sampling_rate = data.fs
    channel_names = data.channels
    seizure_events = annotation_list[rec_idx].events

    for ch_idx, ch_data in enumerate(eeg_data):
        ch_name = channel_names[ch_idx]
        sfreq = sampling_rate[ch_idx]

        # Filter original signal
        bandpassed = bandpass_filter(ch_data, sfreq)
        filtered = notch_filter(bandpassed, sfreq)

        # Segment full-length epochs
        epochs = segment_epochs(filtered, sfreq, EPOCH_DURATION)
        
        for i, epoch in enumerate(epochs):
            #label = is_seizure(i * EPOCH_DURATION, (i + 1) * EPOCH_DURATION, seizure_events)
            for signal_variant, aug_type in [(epoch, "original"), (generate_ft_surrogate(epoch), "ft_surrogate")]:
                # Segment into 1s parts
                one_sec_segments = segment_epochs(signal_variant, sfreq, SECOND_DURATION)

                for sec_idx, segment in enumerate(one_sec_segments):
                    start_time = (i * EPOCH_DURATION) + sec_idx
                    end_time = start_time + 1
                    label = is_seizure(start_time, end_time, seizure_events)

                    f_stat = extract_statistical_features(segment)
                    f_temp = extract_temporal_features(segment)
                    f_comp = extract_complexity_features(segment)
                    f_spec = extract_spectral_features(segment, sfreq)

                    feature = {
                        "recording": rec_idx + 1,
                        "channel": ch_name,
                        "epoch": i + 1,
                        "second": sec_idx + 1,
                        "augmentation": aug_type,
                        "seizure": label,
                        **f_stat, **f_temp, **f_comp, **f_spec
                    }
                    all_features.append(feature)

# Convert and save
df = pd.DataFrame(all_features)
df.to_csv("../datasets/eeg_features_baseline.csv", index=False)
print("SeizFt feature extraction completed!")
