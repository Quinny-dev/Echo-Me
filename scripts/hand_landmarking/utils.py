# scripts/hand_landmarking/utils.py
"""
Drawing helpers and MediaPipe → canonical conversion utilities.
Canonical format: left slot (21×3), then right slot (21×3) -> flattened 126 vector.
Also returns presence mask: np.array([left_present, right_present], dtype=np.int8)

This version is robust to differences between MediaPipe releases (style function names).
"""

import numpy as np
import cv2
import mediapipe as mp
from typing import Tuple, Optional

# Initialize MediaPipe drawing and hands solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def _get_landmark_style():
    """
    Get drawing style for hand landmarks, compatible across MediaPipe versions.
    Some MP versions use plural 'landmarks_style', others use singular 'landmark_style'.
    Returns None if style function is not available, allowing fallback to defaults.
    """
    # Try plural version first
    fn = getattr(mp_drawing_styles, "get_default_hand_landmarks_style", None)
    if fn is None:
        # Fall back to singular version
        fn = getattr(mp_drawing_styles, "get_default_hand_landmark_style", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None

def _get_connection_style():
    """
    Get drawing style for hand connections, compatible across MediaPipe versions.
    Similar to landmark style, handles plural/singular naming differences.
    """
    # Try plural version first
    fn = getattr(mp_drawing_styles, "get_default_hand_connections_style", None)
    if fn is None:
        # Fall back to singular version
        fn = getattr(mp_drawing_styles, "get_default_hand_connection_style", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None

# Cache drawing styles at module load time for better performance
_DEFAULT_LANDMARK_STYLE = _get_landmark_style()
_DEFAULT_CONNECTION_STYLE = _get_connection_style()

def draw_landmarks(frame: np.ndarray, hand_landmarks_list):
    """
    Draw MediaPipe hand landmarks and connections on a BGR frame.
    
    Args:
        frame: BGR image as numpy array
        hand_landmarks_list: List of hand landmarks from MediaPipe results.multi_hand_landmarks
    
    Returns:
        Frame with landmarks drawn on it
    """
    if hand_landmarks_list is None:
        return frame

    # Draw landmarks for each detected hand
    for hand_landmarks in hand_landmarks_list:
        try:
            # Use cached styles if available, otherwise MediaPipe uses defaults
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                _DEFAULT_LANDMARK_STYLE,
                _DEFAULT_CONNECTION_STYLE,
            )
        except Exception:
            # Fallback: draw without style specifications
            try:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            except Exception:
                # Skip this hand if drawing fails completely
                continue
    return frame

def mp_results_to_canonical_two_hand_vector(mp_results, image_shape: Optional[tuple] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert MediaPipe hand detection results to standardized format.
    
    Args:
        mp_results: MediaPipe results object containing hand landmarks
        image_shape: Optional (h, w, c) tuple to convert normalized coords to pixels
    
    Returns:
        Tuple of:
        - flat: 126-element array (left hand 63 coords + right hand 63 coords)
        - presence: 2-element array [left_present, right_present] as 0/1 flags
    
    Each hand has 21 landmarks with (x, y, z) coordinates = 63 values total.
    Left hand slot comes first, then right hand slot in the flattened array.
    """
    # Initialize empty slots for left and right hands (21 landmarks × 3 coords each)
    slot_left = np.zeros((21, 3), dtype=np.float32)
    slot_right = np.zeros((21, 3), dtype=np.float32)
    present = np.array([0, 0], dtype=np.int8)  # [left_present, right_present]

    # Return empty results if no MediaPipe results
    if mp_results is None:
        return slot_left.reshape(-1).astype(np.float32), present

    # Extract landmarks and handedness from MediaPipe results
    landmarks_list = getattr(mp_results, "multi_hand_landmarks", None)
    handedness_list = getattr(mp_results, "multi_handedness", None)

    # Return empty results if no hands detected
    if not landmarks_list:
        return np.concatenate([slot_left.reshape(-1), slot_right.reshape(-1)]).astype(np.float32), present

    # Process each detected hand
    for i, hand_landmarks in enumerate(landmarks_list):
        # Determine if this is left or right hand from MediaPipe classification
        label = None
        if handedness_list and len(handedness_list) > i:
            try:
                label_str = handedness_list[i].classification[0].label  # "Left" or "Right"
                label = label_str.lower()[0]  # Convert to 'l' or 'r'
            except Exception:
                label = None

        # Extract landmark coordinates (x, y, z) for all 21 points
        coords = []
        for lm in hand_landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])
        coords = np.asarray(coords, dtype=np.float32)  # Shape: (21, 3)

        # Convert normalized coordinates to pixel coordinates if image shape provided
        if image_shape is not None:
            h, w = image_shape[0], image_shape[1]
            coords_px = coords.copy()
            coords_px[:, 0] = coords[:, 0] * w  # Scale x coordinates
            coords_px[:, 1] = coords[:, 1] * h  # Scale y coordinates
            coords = coords_px

        # Assign coordinates to left or right slot based on hand classification
        if label == 'l':
            slot_left = coords
            present[0] = 1
        elif label == 'r':
            slot_right = coords
            present[1] = 1
        else:
            # Fallback: use average x position to determine left/right
            xs_mean = coords[:, 0].mean()
            if image_shape is None:
                center_threshold = 0.5  # For normalized coordinates
            else:
                center_threshold = image_shape[1] / 2.0  # For pixel coordinates
            
            if xs_mean < center_threshold:
                slot_left = coords
                present[0] = 1
            else:
                slot_right = coords
                present[1] = 1

    # Flatten and concatenate left and right hand coordinates
    flat = np.concatenate([slot_left.reshape(-1), slot_right.reshape(-1)]).astype(np.float32)
    return flat, present

def count_detected_hands(flat_landmarks) -> int:
    """
    Count the number of hands present in a flattened landmark vector.
    
    Args:
        flat_landmarks: 126-element array with layout [left_hand_coords(63), right_hand_coords(63)]
                       Missing hands are represented as zeros
    
    Returns:
        Number of detected hands (0, 1, or 2)
    """
    if flat_landmarks is None:
        return 0
        
    # Convert to list and ensure exactly 126 elements
    arr = list(flat_landmarks)
    if len(arr) < 126:
        arr = arr + [0.0] * (126 - len(arr))  # Pad with zeros
    elif len(arr) > 126:
        arr = arr[:126]  # Truncate to 126 elements

    # Split into left hand (first 63) and right hand (last 63) coordinates
    left = arr[:63]
    right = arr[63:126]

    # Check if each hand has non-zero coordinates (threshold for floating point comparison)
    left_present = any(abs(x) > 1e-6 for x in left)
    right_present = any(abs(x) > 1e-6 for x in right)
    
    return int(left_present) + int(right_present)
