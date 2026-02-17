import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.fft import rfft
from scipy.stats import entropy


TARGET_FS = 100         
WINDOW_SEC = 3
STRIDE_SEC = 1


def resample_signal(accel, orig_fs, target_fs=100):
    if orig_fs == target_fs:
        return accel

    duration = accel.shape[0] / orig_fs
    target_len = int(duration * target_fs)

    resampled = resample(accel, target_len, axis=0)
    return resampled

def sliding_windows(signal, fs, window_sec=3, stride_sec=1):
    win_len = int(window_sec * fs)
    stride = int(stride_sec * fs)

    windows = []
    timestamps = []

    for start in range(0, len(signal) - win_len, stride):
        end = start + win_len
        windows.append(signal[start:end])
        timestamps.append(start / fs)  # seconds

    return np.array(windows), np.array(timestamps)

def compute_features(window, fs):
    """
    window: shape (T, 3) for ax, ay, az
    """
    feats = []

    ax, ay, az = window[:,0], window[:,1], window[:,2]
    mag = np.sqrt(ax**2 + ay**2 + az**2)

    # time-domain stats
    for sig in [ax, ay, az]:
        feats.extend([
            np.mean(sig),
            np.std(sig),
            np.sqrt(np.mean(sig**2)), # RMS
            np.min(sig),
            np.max(sig)
        ])

    # magnitude stats
    feats.extend([
        np.mean(mag),
        np.std(mag),
        np.sqrt(np.mean(mag**2))
    ])

    # jerk
    jerk = np.diff(mag)
    feats.append(np.sum(jerk**2))

    # signal magnitude area
    feats.append(np.sum(np.abs(ax) + np.abs(ay) + np.abs(az)) / len(window))

    # frequency domain
    fft_vals = np.abs(rfft(mag))
    freqs = np.fft.rfftfreq(len(mag), d=1/fs)

    dominant_freq = freqs[np.argmax(fft_vals)]
    feats.append(dominant_freq)

    # spectral entropy
    psd = fft_vals / np.sum(fft_vals)
    feats.append(entropy(psd))

    return np.array(feats)

def get_accel_columns(df):
    # possible column naming schemes
    schemes = [
        ['ax', 'ay', 'az'],
        ['Acc AP', 'Acc ML', 'Acc V'],
        ['acc_x', 'acc_y', 'acc_z'],
        ['x', 'y', 'z']
    ]

    for cols in schemes:
        if all(c in df.columns for c in cols):
            return df[cols].values

    raise ValueError("No valid accelerometer columns found.")

def process_farseeing_file(csv_path, sampling_rate):
    print(f"Processing: {csv_path}")
    print(f"Original FS: {sampling_rate} Hz")

    df = pd.read_csv(csv_path)

    accel = get_accel_columns(df)

    # resample
    accel_resampled = resample_signal(accel, sampling_rate, TARGET_FS)

    # sliding windows
    windows, timestamps = sliding_windows(
        accel_resampled,
        TARGET_FS,
        WINDOW_SEC,
        STRIDE_SEC
    )

    print(f"Total windows: {len(windows)}")

    # compute features
    feature_list = []
    for w in windows:
        feats = compute_features(w, TARGET_FS)
        feature_list.append(feats)

    features = np.vstack(feature_list)

    return features, timestamps
