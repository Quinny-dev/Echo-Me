
from typing import List, Optional
import numpy as np
import os
import uuid

from data_pipeline.utils import export_sequence_json

class SequenceBuffer:
  

    def __init__(self, max_length: int):
        self.max_length = int(max_length)
        self.buffer: List[Optional[np.ndarray]] = []

    def append(self, frame_vector: Optional[List[float]]):
       
        if frame_vector is None:
            self.buffer.append(None)
            # keep only last max_length
            if len(self.buffer) > self.max_length:
                self.buffer = self.buffer[-self.max_length:]
            return

        arr = np.array(frame_vector, dtype=float).reshape(1, -1)
        self.buffer.append(arr)
        if len(self.buffer) > self.max_length:
            self.buffer = self.buffer[-self.max_length:]

    def __len__(self):
        return len(self.buffer)

    def get_sequence(self, pad_value: float = 0.0) -> np.ndarray:
     
        if not self.buffer:
            return np.zeros((self.max_length, 0), dtype=float)

        first_non_none = next((b for b in self.buffer if b is not None), None)
        if first_non_none is None:
            return np.zeros((self.max_length, 0), dtype=float)
        F = first_non_none.shape[1]

        rows = []
        for b in self.buffer:
            if b is None:
                rows.append(np.full((1, F), pad_value, dtype=float))
            else:
                rows.append(np.array(b).reshape(1, F))
        seq = np.concatenate(rows, axis=0)

        if seq.shape[0] < self.max_length:
            pad_amt = self.max_length - seq.shape[0]
            pad = np.full((pad_amt, F), pad_value, dtype=float)
            seq = np.concatenate([pad, seq], axis=0)
        elif seq.shape[0] > self.max_length:
            seq = seq[-self.max_length :]

        return seq.astype(float)

    def save(self, directory: str, prefix: str = "sequence") -> str:
      
        os.makedirs(directory, exist_ok=True)
        seq = self.get_sequence()
        filename = f"{prefix}_{uuid.uuid4().hex}.json"
        path = os.path.join(directory, filename)
        export_sequence_json(seq, path, filename_hint=filename)
        return path
