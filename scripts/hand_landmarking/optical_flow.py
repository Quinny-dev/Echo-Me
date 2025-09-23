# scripts/hand_landmarking/optical_flow.py
"""
Optical-flow helpers to propagate previously-detected landmarks into new frames
when MediaPipe fails or misses a hand. Uses cv2.calcOpticalFlowPyrLK.

Functions:
- track_landmarks_lk(prev_gray, cur_gray, prev_landmarks_xy) -> (propagated_xy, status_mask)
- extract_left_right_from_mp_results(mp_results, image_shape) -> (left_xy,left_z,right_xy,right_z,presence)
- build_flat_from_slots(...) -> flat (126,) and presence flags
"""

import cv2
import numpy as np
from typing import Optional, Tuple

def track_landmarks_lk(prev_gray: np.ndarray, cur_gray: np.ndarray, prev_landmarks_xy: np.ndarray
                       ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Track prev_landmarks_xy (N,2) from prev_gray -> cur_gray using Pyramidal LK.
    Returns:
      p1 (N,2) float32 of tracked points, and st (N,) bool mask where True = tracked successfully.
    If input invalid, returns (None, None).
    """
    if prev_landmarks_xy is None:
        return None, None
    if prev_landmarks_xy.size == 0:
        return None, None
    if prev_gray is None or cur_gray is None:
        return None, None

    p0 = prev_landmarks_xy.astype(np.float32).reshape(-1, 1, 2)
    # parameters tuned for hand motion: larger window and pyramid levels
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None,
                                           winSize=(21, 21), maxLevel=3,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if p1 is None:
        return None, None
    p1 = p1.reshape(-1, 2)
    st = st.reshape(-1).astype(bool)
    return p1, st

def extract_left_right_from_mp_results(mp_results, image_shape: Tuple[int, int, int]):
    """
    Given MediaPipe results object and image_shape (h,w,c), return per-hand pixel coords and z:
      - left_xy (21,2) or None
      - left_z  (21,)  or None
      - right_xy, right_z similarly
      - presence tuple (left_present, right_present)

    If handedness label is missing, assign hand slot by median x (fallback).
    """
    h, w = image_shape[0], image_shape[1]
    left_xy = None; right_xy = None
    left_z = None; right_z = None
    present = (0, 0)

    if mp_results is None:
        return left_xy, left_z, right_xy, right_z, present

    landmarks_list = getattr(mp_results, "multi_hand_landmarks", None)
    handedness_list = getattr(mp_results, "multi_handedness", None)
    if not landmarks_list:
        return left_xy, left_z, right_xy, right_z, present

    for i, hand_landmarks in enumerate(landmarks_list):
        # coords normalized (x,y in [0,1]) and z relative
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)  # (21,3)
        # convert to pixel coords for x,y
        px = coords.copy()
        px[:, 0] = px[:, 0] * w
        px[:, 1] = px[:, 1] * h
        zs = coords[:, 2].copy()

        label = None
        if handedness_list and len(handedness_list) > i:
            try:
                label_str = handedness_list[i].classification[0].label
                label = label_str.lower()[0]  # 'l' or 'r'
            except Exception:
                label = None

        if label == 'l':
            left_xy = px[:, :2].copy()
            left_z = zs.copy()
            present = (1, present[1])
        elif label == 'r':
            right_xy = px[:, :2].copy()
            right_z = zs.copy()
            present = (present[0], 1)
        else:
            # fallback: assign by mean x
            mean_x = px[:, 0].mean()
            if mean_x < (w / 2.0):
                left_xy = px[:, :2].copy()
                left_z = zs.copy()
                present = (1, present[1])
            else:
                right_xy = px[:, :2].copy()
                right_z = zs.copy()
                present = (present[0], 1)

    return left_xy, left_z, right_xy, right_z, present

def build_flat_from_slots(left_xy, left_z, right_xy, right_z, image_shape: Tuple[int,int,int], normalize: bool=True):
    """
    Build the canonical flattened vector (126,) from per-slot pixel coords + z arrays.
    If normalize=True, x/w and y/h are converted back to [0,1] normalized coords.
    For missing slots (None), zeros are used.

    Returns:
      flat (126,) float32
      presence (2,) int8 where 1=slot present
    """
    h, w = image_shape[0], image_shape[1]
    def make_slot(xy, z):
        if xy is None or z is None:
            return np.zeros((21,3), dtype=np.float32)
        slot = np.zeros((21,3), dtype=np.float32)
        if normalize:
            slot[:, 0] = xy[:, 0] / float(w)
            slot[:, 1] = xy[:, 1] / float(h)
        else:
            slot[:, 0] = xy[:, 0]
            slot[:, 1] = xy[:, 1]
        slot[:, 2] = z
        return slot

    slot_left = make_slot(left_xy, left_z)
    slot_right = make_slot(right_xy, right_z)

    presence = np.array([0,0], dtype=np.int8)
    # presence if any non-zero coordinate exists
    if left_xy is not None and np.any(left_xy != 0):
        presence[0] = 1
    if right_xy is not None and np.any(right_xy != 0):
        presence[1] = 1

    flat = np.concatenate([slot_left.reshape(-1), slot_right.reshape(-1)]).astype(np.float32)
    return flat, presence
