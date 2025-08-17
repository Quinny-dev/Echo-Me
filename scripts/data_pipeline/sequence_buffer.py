# scripts/data_pipeline/sequence_buffer.py
"""
SequenceBuffer: fixed-length rolling buffer for 126-dim frame vectors.
"""

from typing import List, Optional
import numpy as np
import os
import uuid

class SequenceBuffer:
    def __init__(self, max_length: int):
        self.max_length = int(max_length)
        self.buffer: List[np.ndarray] = []

    def append(self, frame_vector: Optional[List[float]]):
        """
        Append a single frame vector (expected length 126). If None, append zeros.
        """
        if frame_vector is None:
            vec = np.zeros((126,), dtype=np.float32)
        else:
            vec = np.asarray(frame_vector, dtype=np.float32).flatten()
            if vec.size < 126:
                pad = np.zeros((126 - vec.size,), dtype=np.float32)
                vec = np.concatenate([vec, pad])
            elif vec.size > 126:
                vec = vec[:126]
        self.buffer.append(vec)
        if len(self.buffer) > self.max_length:
            self.buffer = self.buffer[-self.max_length:]

    def get_sequence(self, pad_front: bool = True) -> np.ndarray:
        """
        Return array shaped (max_length, 126). Pad with zeros if not full.
        """
        if len(self.buffer) == 0:
            return np.zeros((self.max_length, 126), dtype=np.float32)
        seq = np.vstack(self.buffer)
        T = seq.shape[0]
        if T < self.max_length:
            pad_amt = self.max_length - T
            pad = np.zeros((pad_amt, seq.shape[1]), dtype=seq.dtype)
            if pad_front:
                seq = np.vstack((pad, seq))
            else:
                seq = np.vstack((seq, pad))
        return seq

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def save(self, directory: str, prefix: str = "sequence"):
        """
        Save the padded sequence to directory as an .npy file; returns path.
        """
        os.makedirs(directory, exist_ok=True)
        seq = self.get_sequence()
        filename = f"{prefix}_{uuid.uuid4().hex}.npy"
        path = os.path.join(directory, filename)
        np.save(path, seq)
        return path
