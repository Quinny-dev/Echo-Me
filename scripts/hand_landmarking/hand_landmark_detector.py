"""
This module uses MediaPipe to detect up to 2 hands in a frame
and returns 126 values (2 hands × 21 landmarks × 3 coordinates).

If fewer than 2 hands are detected, zeros fill the missing values.

Designed to be used as a frame-by-frame hand detector.
Multi-frame sequences are handled in a separate module.
"""

import cv2                          # OpenCV for image processing.
import mediapipe as mp              # MediaPipe for hand tracking.
import numpy as np                  # NumPy for numerical operations.

# Initialize MediaPipe's hand module.
mp_hands = mp.solutions.hands

class HandLandmarkDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2):
        """
        Initializes the MediaPipe Hands model with optional settings.
        :param static_image_mode: True for static images, False for video stream.
        :param max_num_hands: Maximum number of hands to detect per frame (default: 2).
        """
        self.hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_landmarks(self, frame):
        """
        Detects hand landmarks in a single frame.
        Returns a list of 126 float values (2 hands × 21 landmarks × 3 coords).
        If fewer than 2 hands are found, fills with zeros.
        
        :param frame: BGR image from OpenCV.
        :return: Flattened list of 126 float values.
        """
        # Convert frame to RGB as required by MediaPipe.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run the hand landmark detection model.
        results = self.hands.process(rgb)

        # Prepare a default zero-filled buffer for up to 2 hands.
        frame_vector = np.zeros((2, 21, 3), dtype=np.float32)

        # If hands are detected, populate the buffer.
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                for j, lm in enumerate(hand_landmarks.landmark):
                    frame_vector[i, j] = [lm.x, lm.y, lm.z]

        # Return flattened vector: 2 × 21 × 3 = 126 values.
        return frame_vector.flatten().tolist()

    def close(self):
        """Releases MediaPipe resources."""
        self.hands.close()