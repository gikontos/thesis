import numpy as np
from helpers.functions import (
    bandpass_filter, notch_filter, segment_epochs,
    extract_statistical_features, extract_temporal_features,
    extract_complexity_features, compute_power_features,
    compute_theta_beta_ratio
)

class SequentialGenerator:
    def __init__(self, recording, verbose=False):
        self.recording = recording
        self.verbose = verbose
        self.features = self._extract_features()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def _extract_features(self):
        SECOND_DURATION = 1
        sfreq = self.recording.fs[0]

        # Preprocess all channels and crop to minimum length
        min_len = min(len(ch) for ch in self.recording.data)
        preprocessed = []
        for ch_data in self.recording.data:
            ch_data = ch_data[:min_len]
            bandpassed = bandpass_filter(ch_data, sfreq)
            filtered = notch_filter(bandpassed, sfreq)
            preprocessed.append(filtered)

        multi_ch_data = np.stack(preprocessed)  # shape: (n_channels, n_samples)
        n_channels, n_samples = multi_ch_data.shape

        # Segment full data into 1-second slices
        segment_len = int(SECOND_DURATION * sfreq)
        total_segments = n_samples // segment_len

        all_features = []

        for i in range(total_segments):
            segment = multi_ch_data[:, i*segment_len:(i+1)*segment_len]

            per_channel_features = []

            for ch_idx in range(n_channels):
                ch_segment = segment[ch_idx]

                f_stat = extract_statistical_features(ch_segment)
                f_temp = extract_temporal_features(ch_segment)
                f_comp = extract_complexity_features(ch_segment)

                power_dicts = compute_power_features([ch_segment], sfreq)
                theta_beta_list = compute_theta_beta_ratio([ch_segment], sfreq)

                power = power_dicts[0]  # single segment
                theta_beta = theta_beta_list[0]

                feature_dict = {
                    "theta_beta_ratio": theta_beta,
                    **f_stat, **f_temp, **f_comp, **power
                }

                per_channel_features.append(feature_dict)

            # Average each feature across channels
            keys = per_channel_features[0].keys()
            averaged = {k: float(np.mean([ch[k] for ch in per_channel_features])) for k in keys}
            all_features.append(averaged)

        if all_features:
            keys = list(all_features[0].keys())
            return np.array([[f[k] for k in keys] for f in all_features], dtype=np.float32)
        else:
            return np.empty((0, 1), dtype=np.float32)
