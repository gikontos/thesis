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
        SECOND_DURATION = 1
        sfreq = self.recording.fs[0]

        # Make sure all channels are same length
        min_len = min(len(ch) for ch in self.recording.data)
        n_channels = len(self.recording.data)

        # Preprocess all channels
        all_channels = []
        for ch_idx, ch_data in enumerate(self.recording.data):
            ch_data = ch_data[:min_len]
            bandpassed = bandpass_filter(ch_data, sfreq)
            filtered = notch_filter(bandpassed, sfreq)
            all_channels.append(filtered)

        # Stack to (n_channels, n_samples)
        multi_ch_data = np.stack(all_channels)

        # Segment into 58s epochs → (optional, not used below)
        total_samples = multi_ch_data.shape[1]
        sec_samples = int(SECOND_DURATION * sfreq)
        total_secs = total_samples // sec_samples

        all_features = []

        for i in range(total_secs):
            segment = multi_ch_data[:, i*sec_samples:(i+1)*sec_samples]

            ch_features = []
            for ch_idx in range(n_channels):
                ch_seg = segment[ch_idx:ch_idx+1]  # Shape (1, sec_samples)

                # These functions expect (n_epochs, n_samples)
                bands = extract_frequency_bands(ch_seg, sfreq)
                powers = compute_power_features(ch_seg, sfreq)
                theta_beta = compute_theta_beta_ratio(ch_seg, sfreq)
                alpha_peak, one_over_f = compute_fooof_features(ch_seg, sfreq)
                dfa = compute_dfa(ch_seg)

                # Merge features
                feat_dict = {
                    "theta_beta_ratio": theta_beta[0],
                    "one_over_f": one_over_f[0],
                    "dfa_exponent": dfa[0],
                    # "alpha_peak": alpha_peak[0],  # Optional
                }
                feat_dict.update(powers[0])  # Add band powers
                ch_features.append(feat_dict)

            # Average features across channels
            keys = ch_features[0].keys()
            avg_features = {k: float(np.mean([ch[k] for ch in ch_features])) for k in keys}
            all_features.append(np.array(list(avg_features.values()), dtype=np.float32))

        return all_features
