import librosa
import numpy as np
from scipy.signal import butter, filtfilt

def load_audio(audio_path, sr=11025):
    return librosa.load(audio_path, sr=sr)

def bandpass_filter(data, lowcut, highcut, sr, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def detect_global_key(y, sr):
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    avg_chroma = np.mean(chroma, axis=1)
    key_index = np.argmax(avg_chroma)
    return librosa.midi_to_note(key_index + 24)

def detect_tempo(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    return float(tempo), beats

def detect_time_signature(y, sr):
    tempo, beats = detect_tempo(y, sr)

    if beats is None or len(beats) < 4:
        return "Unknown"

    # Convert beats to times in seconds
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # Calculate differences between consecutive beats
    intervals = np.diff(beat_times)

    # Create a histogram of intervals
    histogram, bin_edges = np.histogram(intervals, bins=10, density=True)

    # Find the most common interval
    most_common_interval = bin_edges[np.argmax(histogram)]

    # Calculate the expected interval for one beat based on the tempo
    expected_interval = 60 / tempo

    # Check if the most common interval matches the expected interval
    if np.isclose(most_common_interval, expected_interval, atol=0.05):
        return "4/4 or Common Time"

    # If the interval is approximately 1.5 times the expected interval, it might be 3/4
    elif np.isclose(most_common_interval, expected_interval * 1.5, atol=0.05):
        return "3/4"

    # If the interval is approximately half the expected interval, it might be 6/8 or 12/8
    elif np.isclose(most_common_interval, expected_interval / 2, atol=0.05):
        return "6/8 or 12/8"

    # Otherwise, return unusual time signature
    else:
        return f"Unusual (most_common_interval={most_common_interval:.2f}, expected_interval={expected_interval:.2f})"
