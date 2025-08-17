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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def _get_landmark_style():
    """
    Return a drawing spec object for landmarks if available across MP versions.
    Some MP versions expose get_default_hand_landmarks_style (plural),
    others get_default_hand_landmark_style (singular).
    If not available, return None so draw_landmarks falls back to default.
    """
    fn = getattr(mp_drawing_styles, "get_default_hand_landmarks_style", None)
    if fn is None:
        fn = getattr(mp_drawing_styles, "get_default_hand_landmark_style", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None

def _get_connection_style():
    """
    Similar robust lookup for connection style functions.
    """
    fn = getattr(mp_drawing_styles, "get_default_hand_connections_style", None)
    if fn is None:
        fn = getattr(mp_drawing_styles, "get_default_hand_connection_style", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None

# Precompute style objects (fast path)
_DEFAULT_LANDMARK_STYLE = _get_landmark_style()
_DEFAULT_CONNECTION_STYLE = _get_connection_style()

def draw_landmarks(frame: np.ndarray, hand_landmarks_list):
    """
    Draw mediapipe landmarks on BGR frame. hand_landmarks_list expected to be
    results.multi_hand_landmarks (an iterable) or None.

    This function uses whichever drawing style helpers are available for the
    installed Mediapipe version, and falls back to plain drawing if necessary.
    """
    if hand_landmarks_list is None:
        return frame

    for hand_landmarks in hand_landmarks_list:
        try:
            # If either style spec is None, draw_landmarks accepts None and uses defaults.
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                _DEFAULT_LANDMARK_STYLE,
                _DEFAULT_CONNECTION_STYLE,
            )
        except Exception:
            # As a last-resort fallback, call draw_landmarks without style objects.
            try:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            except Exception:
                # If even that fails, silently continue (do not crash the collector)
                continue
    return frame

def mp_results_to_canonical_two_hand_vector(mp_results, image_shape: Optional[tuple] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a MediaPipe Hands 'results' object (for one frame) into:
      - flat: np.ndarray shape (126,) (left slot first, right slot second)
      - presence: np.ndarray shape (2,) ints [left_present, right_present]

    image_shape: optional (h, w, c); if provided the normalized coordinates will
    be converted to pixel coordinates for x,y (z left as-is). If None, normalized
    coords [0,1] are kept.

    MediaPipe provides results.multi_hand_landmarks and results.multi_handedness
    which are parallel lists. We use handedness.classification[0].label to map to left/right.
    """
    slot_left = np.zeros((21, 3), dtype=np.float32)
    slot_right = np.zeros((21, 3), dtype=np.float32)
    present = np.array([0, 0], dtype=np.int8)

    if mp_results is None:
        return slot_left.reshape(-1).astype(np.float32), present

    landmarks_list = getattr(mp_results, "multi_hand_landmarks", None)
    handedness_list = getattr(mp_results, "multi_handedness", None)

    if not landmarks_list:
        return np.concatenate([slot_left.reshape(-1), slot_right.reshape(-1)]).astype(np.float32), present

    for i, hand_landmarks in enumerate(landmarks_list):
        # get label 'Left' or 'Right' if available
        label = None
        if handedness_list and len(handedness_list) > i:
            try:
                label_str = handedness_list[i].classification[0].label  # e.g. "Left"
                label = label_str.lower()[0]  # 'l' or 'r'
            except Exception:
                label = None

        coords = []
        for lm in hand_landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])
        coords = np.asarray(coords, dtype=np.float32)  # (21,3)

        # Optionally convert normalized to pixel coords
        if image_shape is not None:
            h, w = image_shape[0], image_shape[1]
            coords_px = coords.copy()
            coords_px[:, 0] = coords[:, 0] * w
            coords_px[:, 1] = coords[:, 1] * h
            coords = coords_px

        if label == 'l':
            slot_left = coords
            present[0] = 1
        elif label == 'r':
            slot_right = coords
            present[1] = 1
        else:
            # fallback: use mean x position to decide
            xs_mean = coords[:, 0].mean()
            if image_shape is None:
                center_threshold = 0.5
            else:
                center_threshold = image_shape[1] / 2.0
            if xs_mean < center_threshold:
                slot_left = coords
                present[0] = 1
            else:
                slot_right = coords
                present[1] = 1

    flat = np.concatenate([slot_left.reshape(-1), slot_right.reshape(-1)]).astype(np.float32)
    return flat, present

def count_detected_hands(flat_landmarks) -> int:
    """
    Count how many hands are present in a flattened 126-length landmark vector.
    Expects layout: [left_slot(63 floats), right_slot(63 floats)] where missing hands are zeros.
    Returns 0, 1, or 2.
    """
    if flat_landmarks is None:
        return 0
    arr = list(flat_landmarks)
    # pad/truncate to 126
    if len(arr) < 126:
        arr = arr + [0.0] * (126 - len(arr))
    elif len(arr) > 126:
        arr = arr[:126]

    left = arr[:63]
    right = arr[63:126]

    left_present = any(abs(x) > 1e-6 for x in left)
    right_present = any(abs(x) > 1e-6 for x in right)
    return int(left_present) + int(right_present)
