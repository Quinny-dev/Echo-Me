"""
Minimal HandLandmarkDetector
- Uses MediaPipe to process frames and return raw mediapipe results.
- No flattening; use `.process(frame)` for drawing landmarks or extracting info.
"""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

class HandLandmarkDetector:
    def __init__(self,
                 static_image_mode: bool = True,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """Initialize MediaPipe Hands."""
        self.hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame_bgr):
        """
        Process a BGR frame and return raw MediaPipe results.
        Use `results.multi_hand_landmarks` to draw or analyze hands.
        """
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.hands.process(image_rgb)

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()