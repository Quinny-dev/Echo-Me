import os
import numpy as np
import json
from pathlib import Path

SEQUENCE_LENGTH = 60               # Number of frames per sign.
FPS = 10                          # Capture rate (frames/sec).
HAND_CONNECTIONS = 2                # Allow 1 or 2 hands.
USE_STATIC_IMAGE_MODE = False
DATA_DIR = "data/raw_sequences"
 
def create_dir_if_not_exists(path):
    
    if not os.path.exists(path):
        os.makedirs(path)

def save_sequence_to_disk(sequence: np.ndarray, label: str, base_dir=DATA_DIR):
   
    create_dir_if_not_exists(base_dir)
    label_dir = os.path.join(base_dir, label)
    create_dir_if_not_exists(label_dir)
    # count existing files to generate next index (keeps stable filenames like 1.json, 2.json, ...)
    count = len([f for f in os.listdir(label_dir) if f.lower().endswith('.json')])
    filename = os.path.join(label_dir, f"{count+1}.json")
    # Use export_sequence_json to write correct format
    export_sequence_json(sequence, filename, filename_hint=filename)

def load_sequence_from_disk(path):
    
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

def export_sequence_json(sequence: np.ndarray, out_json_path: str, filename_hint: str = "") -> str:
    
    if sequence is None:
        raise ValueError("sequence is None")

    seq = np.array(sequence)
    if seq.ndim == 1:
        try:
            seq = np.vstack(seq)
        except Exception as e:
            raise ValueError("sequence must be 2D (T,F)") from e

    if seq.ndim != 2:
        raise ValueError(f"sequence must be 2D (T,F), got ndim={seq.ndim}")

    T, F = seq.shape
    items = []

    def to_py(a: np.ndarray):
        return a.astype(float).tolist()

    # 126 => assume left(63) then right(63) in repo; user wants right then left in JSON
    if F == 126:
        left = seq[:, :63]
        right = seq[:, 63:126]
        items.append({"label": "right", "sequence": to_py(right)})
        items.append({"label": "left",  "sequence": to_py(left)})
    elif F == 63:
        hint = (filename_hint or "").lower()
        if "left" in hint:
            items.append({"label": "left", "sequence": to_py(seq)})
        elif "right" in hint:
            items.append({"label": "right", "sequence": to_py(seq)})
        else:
            # default to right to match your example ordering
            items.append({"label": "right", "sequence": to_py(seq)})
    else:
        # try chunking into 63-blocks if possible
        if F % 63 == 0:
            n_chunks = F // 63
            for i in range(n_chunks):
                chunk = seq[:, i*63:(i+1)*63]
                if i == 0:
                    items.append({"label": "right", "sequence": to_py(chunk)})
                elif i == 1:
                    items.append({"label": "left", "sequence": to_py(chunk)})
                else:
                    items.append({"label": f"hand{i+1}", "sequence": to_py(chunk)})
        else:
            # fallback: write whole sequence as 'right' so trainer still receives data
            items.append({"label": "right", "sequence": to_py(seq)})

    out_path = Path(out_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    return str(out_path)


