# This utility provides drawing functions to visualize hand landmarks.

import cv2
import mediapipe as mp

# MediaPipe drawing utility and predefined hand connections.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def draw_landmarks(frame, hand_landmarks_list):
    """
    Draws hand landmarks on the input frame using MediaPipe's drawing utilities.

    Parameters:
    - frame: The input BGR image (OpenCV format).
    - hand_landmarks_list: List of hand landmark objects returned by MediaPipe.

    Returns:
    - Annotated frame with hand landmarks drawn.
    """
    if not hand_landmarks_list:
        return frame  # Return unmodified frame if no hands.

    for hand_landmarks in hand_landmarks_list:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,  # Draw connections between joints.
            mp_drawing_styles.get_default_hand_landmarks_style(),  # Joints.
            mp_drawing_styles.get_default_hand_connections_style()                    # Lines.
        )
    return frame

def count_detected_hands(landmarks):
    """
    Counts how many hands are detected based on landmark data.
    Each hand corresponds to 63 float values (21 landmarks × 3 coords).
    Zeros represent no hand detected.

    :param landmarks: List of 126 float values from the detector.
    :return: Integer count of detected hands (0, 1 or 2).
    """
    hand1 = landmarks[:63]
    hand2 = landmarks[63:]

    hand1_detected = any(v != 0.0 for v in hand1)
    hand2_detected = any(v != 0.0 for v in hand2)

    return int(hand1_detected) + int(hand2_detected)