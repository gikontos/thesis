import numpy as np
from helpers.functions import (
    bandpass_filter, notch_filter, segment_epochs,
    extract_frequency_bands, compute_power_features,
    compute_theta_beta_ratio, compute_fooof_features,
    compute_dfa
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
        all_features = []

        for ch_idx, ch_data in enumerate(self.recording.data):
            sfreq = self.recording.fs[ch_idx]

            # Preprocessing
            bandpassed = bandpass_filter(ch_data, sfreq)
            filtered = notch_filter(bandpassed, sfreq)

            # Segment into 58s epochs
            epochs = segment_epochs(filtered, sfreq, EPOCH_DURATION)

            # Feature extraction
            bands = extract_frequency_bands(epochs, sfreq)
            powers = compute_power_features(epochs, sfreq)
            theta_beta = compute_theta_beta_ratio(epochs, sfreq)
            alpha_peak, one_over_f = compute_fooof_features(epochs, sfreq)
            dfa = compute_dfa(epochs)

            for i in range(len(epochs)):
                feat = {
                    "theta_beta_ratio": theta_beta[i],
                    "alpha_peak": alpha_peak[i],
                    "one_over_f": one_over_f[i],
                    "dfa_exponent": dfa[i],
                }
                feat.update(powers[i])  # Merge band powers

                # Flatten feature vector
                all_features.append(np.array(list(feat.values()), dtype=np.float32))

        return all_features
