import numpy as np
from helpers.functions import (
    bandpass_filter, notch_filter, segment_epochs,
    extract_statistical_features, extract_temporal_features,
    extract_complexity_features, extract_spectral_features
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
        EPOCH_DURATION = 58
        SECOND_DURATION = 1
        all_features = []

        for ch_idx, ch_data in enumerate(self.recording.data):
            sfreq = self.recording.fs[ch_idx]

            # Preprocessing
            bandpassed = bandpass_filter(ch_data, sfreq)
            filtered = notch_filter(bandpassed, sfreq)

            # Epoch segmentation
            epochs = segment_epochs(filtered, sfreq, EPOCH_DURATION)

            for i, epoch in enumerate(epochs):
                one_sec_segments = segment_epochs(epoch, sfreq, SECOND_DURATION)

                for sec_idx, segment in enumerate(one_sec_segments):
                    # Extract all feature types
                    f_stat = extract_statistical_features(segment)
                    f_temp = extract_temporal_features(segment)
                    f_comp = extract_complexity_features(segment)
                    f_spec = extract_spectral_features(segment, sfreq)

                    # Combine into single flat vector
                    feature_vector = {**f_stat, **f_temp, **f_comp, **f_spec}
                    all_features.append(feature_vector)

        if all_features:
            keys = list(all_features[0].keys())
            return np.array([[f[k] for k in keys] for f in all_features], dtype=np.float32)
        else:
            return np.empty((0, 1), dtype=np.float32)
