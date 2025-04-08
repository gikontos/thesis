import numpy as np

def get_events(y_pred, recording, threshold=0.5, min_duration=1):
    """
    Convert prediction probabilities to seizure events.

    Args:
        y_pred (np.ndarray): array of probabilities, one per second.
        recording: used for length matching if needed.
        threshold (float): probability threshold to consider a second as seizure.
        min_duration (int): minimum duration of an event in seconds.

    Returns:
        List of [start, end] seizure event times (in seconds).
    """
    seizure_mask = y_pred >= threshold
    events = []
    start = None

    for i, val in enumerate(seizure_mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_duration:
                events.append([start, i])
            start = None

    # Handle case where seizure goes to the end
    if start is not None and len(seizure_mask) - start >= min_duration:
        events.append([start, len(seizure_mask)])

    return events
