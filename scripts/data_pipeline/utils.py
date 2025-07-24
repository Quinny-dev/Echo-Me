import os
import numpy as np

SEQUENCE_LENGTH = 30                # Number of frames per sign.
FPS = 20                            # Capture rate (frames/sec).
HAND_CONNECTIONS = 2                # Allow 1 or 2 hands.
USE_STATIC_IMAGE_MODE = False
DATA_DIR = "data/raw_sequences"

def create_dir_if_not_exists(path):
    """
    Create directory if it doesn't exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_sequence_to_disk(sequence: np.ndarray, label: str, base_dir=DATA_DIR):
    """
    Save a single sequence (numpy array) to disk under label folder with unique filename.
    """
    create_dir_if_not_exists(base_dir)
    label_dir = os.path.join(base_dir, label)
    create_dir_if_not_exists(label_dir)
    count = len(os.listdir(label_dir))
    filename = os.path.join(label_dir, f"{count+1}.npy")
    np.save(filename, sequence)

def load_sequence_from_disk(path):
    """
    Load a numpy sequence from disk.
    """
    return np.load(path)

def normalize_landmarks_single(hand: np.ndarray) -> np.ndarray:
    """
    Center on the wrist (landmark 0) and scale so the
    max wrist→joint distance = 1.
    `hand` is shape (21,3).
    """
    lm = hand.astype(np.float32)
    wrist = lm[0]
    lm -= wrist
    d = np.linalg.norm(lm, axis=1)
    max_d = d.max() if d.max() > 0 else 1.0
    return lm / max_d

def sliding_window_blocks(
    seq: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """
    Break a sequence of shape (T, 2, 21, 3) into overlapping windows.
    Pads at the start with NaNs if T < window_size.
    Returns shape (N_windows, window_size, 2, 21, 3).
    """
    T = seq.shape[0]
    if T < window_size:
        pad_amt = window_size - T
        pad = np.full((pad_amt, *seq.shape[1:]), np.nan, dtype=seq.dtype)
        seq = np.concatenate([pad, seq], axis=0)
        T = window_size

    N = 1 + (T - window_size) // stride
    windows = []
    for i in range(0, stride * N, stride):
        windows.append(seq[i : i + window_size])
    return np.stack(windows, axis=0)
