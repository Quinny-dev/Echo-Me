# scripts/hand_landmarking/hand_landmark_detector.py
"""
HandLandmarkDetector
- Uses MediaPipe to process frames and return mediapipe results.
- Provides compatibility method `detect_landmarks(frame)` used by existing tests,
  which returns a flattened 126-length list: left_slot(63) then right_slot(63).
- Also exposes `process(frame)` to get raw MediaPipe results when handedness is needed.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple

mp_hands = mp.solutions.hands

class HandLandmarkDetector:
    def __init__(self,
                 static_image_mode: bool = True,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Hands.
        """
        self.hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray):
        """
        Run MediaPipe Hands on a BGR OpenCV frame and return the raw MediaPipe result object.
        The caller can read:
          - results.multi_hand_landmarks (list of landmark lists)
          - results.multi_handedness (list of classification objects with label and score)
        """
        # MediaPipe expects RGB images
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        return results

    def detect_flat(self, frame_bgr: np.ndarray) -> list:
        """
        Convenience: process the frame and return a flattened 126-length list
        (left_slot 63 floats followed by right_slot 63 floats). If no hand in a slot,
        that slot is zeros. This method does NOT return handedness labels.
        Use .process(...) if you need handedness information.
        """
        results = self.process(frame_bgr)
        # default zero slots: we will use slot 0 and slot 1 in the order MediaPipe returned
        # (this is just a flat representation; handedness should be resolved elsewhere when needed)
        frame_vector = np.zeros((2, 21, 3), dtype=np.float32)

        if results is None or not getattr(results, "multi_hand_landmarks", None):
            return frame_vector.flatten().tolist()

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            for j, lm in enumerate(hand_landmarks.landmark):
                frame_vector[i, j] = [lm.x, lm.y, lm.z]

        return frame_vector.flatten().tolist()

    def detect_landmarks(self, frame_bgr: np.ndarray, return_results: bool = False) -> Optional[Tuple[list, object]]:
        """
        Compatibility wrapper expected by existing test code.

        By default returns:
          - flat_list : list of 126 floats (left_slot then right_slot) -- same as detect_flat

        If return_results=True, returns a tuple (flat_list, mp_results) where mp_results is the raw MediaPipe result.
        """
        flat = self.detect_flat(frame_bgr)
        if return_results:
            mp_results = self.process(frame_bgr)
            return flat, mp_results
        return flat

    def close(self):
        """
        Release MediaPipe resources.
        """
        self.hands.close()
