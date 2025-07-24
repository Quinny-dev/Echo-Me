import numpy as np

class SequenceBuffer:
    """
    A fixed-length rolling buffer to store sequences of hand landmark frames.
    Each frame is expected to be a list or array of length 126 (2 hands × 21 landmarks × 3 coords).
    """

    def __init__(self, max_length: int):
        """
        Initialize the sequence buffer.
        
        Args:
            max_length (int): Maximum number of frames to hold in the buffer.
        """
        self.max_length = max_length
        self.buffer = []

    def add_frame(self, frame):
        """
        Add a new frame to the buffer. If the buffer exceeds max_length, the oldest frame is removed.
        
        Args:
            frame (list or np.array): A single frame of landmarks (length 126).
        """
        if len(frame) != 126:
            raise ValueError(f"Frame length should be 126, got {len(frame)}")

        self.buffer.append(frame)

        # Maintain fixed size buffer by popping oldest frame if needed
        if len(self.buffer) > self.max_length:
            self.buffer.pop(0)

    def get_sequence(self):
        """
        Get the current sequence of frames as a numpy array.
        If the buffer has fewer frames than max_length, zero-pad at the beginning.
        
        Returns:
            np.ndarray: Array of shape (max_length, 126) with float values.
        """
        seq_len = len(self.buffer)
        if seq_len == 0:
            # Return all zeros if empty
            return np.zeros((self.max_length, 126), dtype=np.float32)
        
        # Stack buffered frames into numpy array
        seq = np.array(self.buffer, dtype=np.float32)

        # Pad with zeros at the front if needed
        if seq_len < self.max_length:
            padding = np.zeros((self.max_length - seq_len, 126), dtype=np.float32)
            seq = np.vstack((padding, seq))

        return seq

    def clear(self):
        """
        Clear the buffer.
        """
        self.buffer = []

    def __len__(self):
        """
        Get current number of frames in buffer.
        """
        return len(self.buffer)