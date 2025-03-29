import numpy as np
import scipy.signal as signal
#from sklearn.decomposition import FastICA
from scipy.signal import welch
import mne
from fooof import FOOOF
from nolds import dfa

# Bandpass filter: 1-100 Hz
def bandpass_filter(data, sfreq, lowcut=1, highcut=100, order=4):
    nyquist = 0.5 * sfreq
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data)

# Notch filter: 50 Hz
def notch_filter(data, sfreq, notch_freq=50):
    nyquist = 0.5 * sfreq
    q_factor = 30  # Quality factor (adjustable)
    b, a = signal.iirnotch(notch_freq / nyquist, q_factor)
    return signal.filtfilt(b, a, data)

# Function to segment data into 1s epochs
def segment_epochs(data, sfreq, epoch_length=1):
    samples_per_epoch = int(sfreq * epoch_length)  # Number of samples per epoch
    num_epochs = len(data) // samples_per_epoch  # Compute number of full epochs
    return np.array(np.split(data[:num_epochs * samples_per_epoch], num_epochs))  # Reshape into epochs

# def apply_fastica(eeg_epochs, n_components=None):
#     """
#     Apply FastICA to remove ocular and ECG artifacts from EEG epochs.
    
#     Parameters:
#     - eeg_epochs: list or np.array of shape (n_epochs, n_samples) containing segmented EEG data.
#     - n_components: Number of independent components. Defaults to the number of channels (automatic).
    
#     Returns:
#     - cleaned_epochs: np.array with cleaned EEG epochs after artifact removal.
#     """
#     eeg_epochs = np.array(eeg_epochs)  # Ensure it's a NumPy array
#     n_epochs, n_samples = eeg_epochs.shape
    
#     # Set number of components equal to channels (for 2-channel EEG, this is 2)
#     if n_components is None:
#         n_components = eeg_epochs.shape[0]  # One component per channel
    
#     # Apply FastICA
#     ica = FastICA(n_components=n_components, random_state=42)
#     sources = ica.fit_transform(eeg_epochs.T).T  # Transpose for proper shape
    
#     # Automatically identify artifact-related components (basic method: high variance)
#     component_variances = np.var(sources, axis=1)
#     artifact_threshold = np.percentile(component_variances, 90)  # Top 10% as artifacts
#     artifact_components = np.where(component_variances > artifact_threshold)[0]  # Indexes of artifacts

#     # Remove artifacts by zeroing out identified components
#     sources[artifact_components, :] = 0
    
#     # Reconstruct EEG signal without artifacts
#     cleaned_epochs = ica.inverse_transform(sources.T).T  # Inverse transform back

#     return cleaned_epochs

def bandpass_filter_custom(data, sfreq, lowcut, highcut, order=4):
    """
    Apply a bandpass filter for a specific frequency range.

    Parameters:
    - data: np.array, EEG signal
    - sfreq: int, Sampling frequency
    - lowcut: float, Lower frequency bound
    - highcut: float, Upper frequency bound
    - order: int, Order of the Butterworth filter

    Returns:
    - filtered_data: np.array, Bandpassed EEG signal
    """
    nyquist = 0.5 * sfreq
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data)


def extract_frequency_bands(eeg_data, sfreq):
    """
    Extract EEG signal into five canonical frequency bands.

    Parameters:
    - eeg_data: np.array of shape (n_samples,)
    - sfreq: int, Sampling frequency

    Returns:
    - bands: dict containing band-filtered signals
    """
    bands = {
        "delta": bandpass_filter_custom(eeg_data, sfreq, 1.25, 4),
        "theta": bandpass_filter_custom(eeg_data, sfreq, 4, 8),
        "alpha": bandpass_filter_custom(eeg_data, sfreq, 8, 13),
        "beta": bandpass_filter_custom(eeg_data, sfreq, 13, 30),
        "gamma": bandpass_filter_custom(eeg_data, sfreq, 30, 49),
    }
    return bands

def compute_power_features(epochs, sfreq):
    """Compute absolute and relative power for different EEG bands using Welch's method."""
    freqs = {
        'delta': (1.25, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 49)
    }
    
    power_features = []

    for epoch in epochs:
        # Welch's method: nperseg should be at least 1/8 of epoch length for stability
        nperseg = max(256, len(epoch) // 8)  # Adjusting to prevent too-short segments
        freq, psd = welch(epoch, sfreq, nperseg=nperseg)

        total_power = np.sum(psd)

        band_powers = {}
        for band, (low, high) in freqs.items():
            idx = np.logical_and(freq >= low, freq <= high)
            band_power = np.sum(psd[idx])
            band_powers[f"{band}_absolute"] = band_power
            band_powers[f"{band}_relative"] = band_power / total_power if total_power > 0 else 0  # Avoid division by zero
        
        power_features.append(band_powers)

    return power_features

# Function to compute theta/beta ratio
def compute_theta_beta_ratio(epochs, sfreq):
    ratios = []
    for epoch in epochs:
        # Adjust the multitaper settings to avoid the low_bias issue
        psd, freq = mne.time_frequency.psd_array_multitaper(epoch, sfreq=sfreq, fmin=1, fmax=100, bandwidth=1, low_bias=False, verbose=False)
        
        # Define frequency bands (theta: 4-8 Hz, beta: 13-30 Hz)
        theta_idx = np.logical_and(freq >= 4, freq <= 8)
        beta_idx = np.logical_and(freq >= 13, freq <= 30)

        theta_power = np.sum(psd[theta_idx])
        beta_power = np.sum(psd[beta_idx])

        theta_beta_ratio = theta_power / beta_power if beta_power > 0 else 0  # Avoid division by zero
        ratios.append(theta_beta_ratio)

    return ratios

def compute_fooof_features(epochs, sfreq):
    alpha_peaks = []
    one_over_f_exponents = []

    for epoch in epochs:
        # Compute the PSD using multitaper method
        psd, freq = mne.time_frequency.psd_array_multitaper(epoch, sfreq=sfreq, fmin=1, fmax=100, bandwidth=1, low_bias=False, verbose=False)
        
        # Initialize the FOOOF object with the correct peak width limits
        fm = FOOOF(peak_width_limits=(2.0, 12.0))  # Set min peak width to 2.0 Hz, max peak width to 12.0 Hz

        # Fit the model to the PSD data
        fm.fit(freq, psd)

        # Get the peak parameters and 1/f exponent
        peak_params = fm.get_params('peak_params')
        aperiodic_params = fm.get_params('aperiodic_params')

        # Check if there are peaks detected
        if isinstance(peak_params, list) and len(peak_params) > 0:
            alpha_peak = peak_params[0][0]  # Extract the frequency of the first peak
        else:
            alpha_peak = None  # No peak detected

        # Extract the 1/f exponent (if available)
        one_over_f = aperiodic_params[1] if len(aperiodic_params) > 1 else None

        # Append the values
        alpha_peaks.append(alpha_peak)
        one_over_f_exponents.append(one_over_f)

    return alpha_peaks, one_over_f_exponents




# Compute DFA exponent (long-range temporal correlations)
def compute_dfa(epochs):
    dfa_exponents = [dfa(epoch) for epoch in epochs]
    return dfa_exponents

def is_seizure(epoch_start, epoch_end, seizure_events):
    """
    Determines if an epoch overlaps with any seizure event.

    :param epoch_start: Start time of the epoch (in seconds)
    :param epoch_end: End time of the epoch (in seconds)
    :param seizure_events: List of (start, end) seizure times
    :return: 1 if the epoch falls in a seizure event, else 0
    """
    for start, end in seizure_events:
        if epoch_start < end and epoch_end > start:  # Overlapping condition
            return 1
    return 0
